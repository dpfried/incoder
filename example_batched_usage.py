from multiprocessing.sharedctypes import Value
import numpy as np
from typing import List

import torch
import tokenizers
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StoppingCriteria
import json

tokenizers_version = tuple(int(n) for n in tokenizers.__version__.split('.'))
if tokenizers_version < (0, 12, 1):
    print("warning: Your tokenizers version looks old and you will likely have formatting issues. We recommend installing tokenizers >= 0.12.1")

PAD = "<pad>"
# signals the start of a document
BOS = "<|endoftext|>"
# signals the end of a generated infill
EOM = "<|endofmask|>"

def make_sentinel(i):
    # signals (1) a location to insert an infill and (2) the start of the infill generation
    return f"<|mask:{i}|>"

def remove_extra_code(input):
    min_stop_position = len(input)
    stop_tokens = ["\nclass", "\ndef", "\n#", "\nif", "\nassert", "\nclass", "<|/ file"]
    for stop_token in stop_tokens:
        if stop_token in input:
            min_stop_position = min(min_stop_position, input.index(stop_token)) 
    return input[:min_stop_position]

# monkey-patch transformers to avoid nans in padded generation with float16
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    # mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min / 10))
    mask = torch.full((tgt_len, tgt_len), torch.tensor(-1e4))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

transformers.models.xglm.modeling_xglm._make_causal_mask = _make_causal_mask

class StopWordsStoppingCriteria(StoppingCriteria):
    def __init__(self, init_lengths: List[int], stop_words_encoded: List[List[int]]):
        super().__init__()
        self.init_lengths = init_lengths
        if stop_words_encoded is None:
            stop_words_encoded = []
        else:
            assert isinstance(stop_words_encoded[0], list)
        assert isinstance(stop_words_encoded, list)
        self.stop_words_encoded = stop_words_encoded

    def _contains_stop_words(self, tokens: List[int]):
        if not bool(self.stop_words_encoded):
            return False
        for start_ix in range(len(tokens)):
            for swe in self.stop_words_encoded:
                if tokens[start_ix:start_ix+len(swe)] == swe:
                    return True
        return False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for init_length, i_tokens in zip(self.init_lengths, input_ids):
            if not self._contains_stop_words(i_tokens[init_length:].tolist()):
                return False
        return True

class InfillingModel:
    def __init__(self, model_name="facebook/incoder-1B", cuda=True, device=None, tokenizer=None, half=True, model=None):
        self.model_name = model_name

        if cuda:
            assert device is None or device.startswith("cuda")
            if device is None:
                device = "cuda"
        else:
            assert device is None or device == "cpu"
            if device is None:
                device = "cpu"
        
        self.device = device

        if model_name == 'facebook/incoder-6B':
            if cuda:
                kwargs = dict(
                    revision="float16", 
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
            else:
                kwargs = dict(
                    low_cpu_mem_usage=True,
                )
        else:
            kwargs = {}

        if model is None:
            print("loading model")
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if tokenizer is None:
            print("loading tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = PAD
        assert self.tokenizer.pad_token_id == 1
        print("loading complete")
        
        if cuda and half:
            self.half = True
            model = model.half()
        else:
            self.half = False
        model = model.to(device)
        self.model = model
        self.cuda = cuda

    def batched_generate(self, inputs: List[str], max_to_generate: int=128, temperature: float=0.2, trim: bool=True, stop_words=None):

        assert self.tokenizer.padding_side == 'left'
        assert isinstance(inputs, list)
        batch = self.tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")
        batch = batch.to(self.device)
        max_input_length = batch.input_ids.size(1)
        max_length = max_input_length + max_to_generate
        stopping_criteria = StoppingCriteriaList()
        if stop_words is not None:
            stop_words_encoded = [self.tokenizer.encode(word, add_special_tokens=False) for word in stop_words]
            stopping_criteria.append(StopWordsStoppingCriteria([max_input_length for l in inputs], stop_words_encoded))
        if max_length > 2048:
            print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
        with torch.no_grad():
            outputs = self.model.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask, do_sample=True, top_p=0.95, temperature=temperature, max_length=max_length, stopping_criteria=stopping_criteria)
        
        hypo_strs = []
        for input, output in zip(inputs, outputs):
            detok_hypo_str = self.tokenizer.decode(output.flatten(), clean_up_tokenization_spaces=False)
            while detok_hypo_str.startswith(PAD):
                detok_hypo_str = detok_hypo_str[len(PAD):]
            if detok_hypo_str.startswith(BOS):
                detok_hypo_str = detok_hypo_str[len(BOS):]

            if trim:
                detok_hypo_str = detok_hypo_str[len(input):]
                detok_hypo_str = remove_extra_code(detok_hypo_str)
            hypo_strs.append(detok_hypo_str)
        return hypo_strs

    def generate(self, input: str, max_to_generate: int=128, temperature: float=0.2, trim: bool=True):
        """
        Do standard left-to-right completion of the prefix `input` by sampling from the model
        """
        outputs = self.batched_generate([input], max_to_generate, temperature, trim)
        assert len(outputs) == 1
        return outputs[0]

    def batched_infill(self, batched_parts: List[List[str]], max_to_generate: int=128, temperature: float=0.2, extra_sentinel: bool=True, max_retries: int=1):
        assert isinstance(batched_parts, list)
        assert isinstance(batched_parts[0], list)
        batch_size = len(batched_parts)
        num_parts = len(batched_parts[0])
        assert all(len(l) == num_parts for l in batched_parts), "all elements in the batch must have the same number of parts"

        # if max_retries > 1 and len(batched_parts) > 1:
        #     raise NotImplementedError("multiple retries with batch > 1")

        # assert num_parts == 2

        batched_retries_attempted = torch.zeros(batch_size).long()
        retries_attempted = 0

        batched_not_done = torch.ones(batch_size).bool()

        done_batched_complete = [None for _ in range(batch_size)]
        done_batched_infills = [None for _ in range(batch_size)]

        while (batched_not_done.any()) and (retries_attempted < max_retries):
            retries_attempted += 1

            batched_infills = [[] for _ in range(batch_size)]
            batched_complete = [[] for _ in range(batch_size)]
            batched_prompts = []

            not_done_indices = batched_not_done.nonzero().flatten()
            batched_retries_attempted[not_done_indices] += 1
            assert batched_retries_attempted.max().item() == retries_attempted

            for parts in batched_parts:
                ## (1) build the prompt
                if len(parts) == 1:
                    prompt = parts[0]
                else:
                    prompt = ""
                    # encode parts separated by sentinel
                    for sentinel_ix, part in enumerate(parts):
                        prompt += part
                        if extra_sentinel or (sentinel_ix < len(parts) - 1):
                            prompt += make_sentinel(sentinel_ix)
                batched_prompts.append(prompt)
            
            ## (2) generate infills
            subbatch_not_done = batched_not_done[not_done_indices].clone()
            assert subbatch_not_done.all()
            subbatch_not_done[:] = False

            for sentinel_ix in range(num_parts - 1):
                batched_part = [parts[sentinel_ix] for parts in batched_parts]
                batched_prompts = [prompt + make_sentinel(sentinel_ix) for prompt in batched_prompts]
                for batch_index, parts in enumerate(batched_parts):
                    batched_complete[batch_index].append(parts[sentinel_ix])

                # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
                subbatch_prompts = [batched_prompts[ix] for ix in not_done_indices]
                subbatch_outputs = self.batched_generate(subbatch_prompts, max_to_generate, temperature, trim=False, stop_words=[EOM])
                for subbatch_ix, (completion, prompt) in enumerate(zip(subbatch_outputs, subbatch_prompts)):
                    batch_ix = not_done_indices[subbatch_ix]
                    completion = completion[len(prompt):]
                    if EOM not in completion:
                        completion += EOM
                        subbatch_not_done[subbatch_ix] |= True
                    completion = completion[:completion.index(EOM) + len(EOM)]
                    infilled = completion[:-len(EOM)]
                    batched_infills[batch_ix].append(infilled)
                    batched_complete[batch_ix].append(infilled)
                    batched_prompts[batch_ix] += completion
            for batch_ix, parts in enumerate(batched_parts):
                batched_complete[batch_ix].append(parts[-1])
            
            batched_not_done[not_done_indices] = subbatch_not_done
            for batch_ix in not_done_indices:
                if not batched_not_done[batch_ix] or retries_attempted >= max_retries:
                    done_batched_complete[batch_ix] = batched_complete[batch_ix]
                    done_batched_infills[batch_ix] = batched_infills[batch_ix]

        done_batched_text = [''.join(complete) for complete in done_batched_complete]

        return [{
            'text': text, # str, the completed document (with infills inserted)
            'parts': parts, # List[str], length N. Same as passed to the method
            'infills': infills, # List[str], length N-1. The list of infills generated
            'retries_attempted': int(this_retries_attempted.item()), # number of retries used (if max_retries > 1)
            'completed': bool(not this_not_done),
        }  for text, parts, infills, this_retries_attempted, this_not_done in zip(
            done_batched_text, batched_parts, done_batched_infills, batched_retries_attempted, batched_not_done
        )]

    def infill(self, parts: List[str], max_to_generate: int=128, temperature: float=0.2, extra_sentinel: bool=True, max_retries: int=1):
        """
        Generate infills to complete a partial document, e.g.
        [A C E] -> [A B C D E], where B and D are infills that have been generated.
        parts: List[str]. list of parts of the document. One string will be
                inserted in between each element, i.e. infilling N-1 locations for a list
                of length N.
        max_to_generate: int. maximum number of tokens to generate. Keep in mind
                that the model context size is 2048.
        temperature: float. temperature parameter for sampling.
        extra_sentinel: bool. we recommend setting this to True, as it makes it
                easier for the model to end generated infills. See the footnote in 
                section 2.2 of our paper for details.
        max_retries: int. if > 1, use rejection sampling to keep sampling infills until
                all infills sample a completion token.
        returns a dictionary containing the following:
            text:  str, the completed document (with infills inserted)
            parts:  List[str], length N. Same as passed to the method
            infills:  List[str], length N-1. The list of infills generated
            retries_attempted:  number of retries used (if max_retries > 1)
        """
        outputs = self.batched_infill([parts], max_to_generate, temperature, extra_sentinel, max_retries)
        assert len(outputs) == 1
        return outputs[0]

infilling_model = InfillingModel("facebook/incoder-1B", cuda=True, half=False)

all_examples =  [
'''\
def count_words(filename):
    """ <insert> """
    counts = Counter()
    with open(filename) as file:
        for line in file:
            words = line.split(' ')
            counts.update(words)
    return counts\
''',
'''\
def count_lines(filename):
    """ <insert> """
    counts = Counter()
    with open(filename) as file:
        return(len(list(file)))\
'''
]

all_parts = [example.split("<insert>") for example in all_examples]

all_results = infilling_model.batched_infill(all_parts, max_to_generate=128, temperature=0.2)

for result in all_results:
  print("completed document:")
  print(result["text"])
  print()
