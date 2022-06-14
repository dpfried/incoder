import os
import time
from unicodedata import bidirectional
import numpy as np
from typing import List, Optional
import random
import sys
import argparse

from collections import namedtuple
try:
    import torch
except:
    print("couldn't import torch; won't be able to use most models", file=sys.stderr)

from utils import truncate_overlap, truncate_num_lines, stripped_line_split, truncate_docstring_infill

DEFAULT_MAX_TOKENS = 450
CODEX_RETRY_DELAY_SECONDS = 60
CODEX_MAX_RETRIES = 30

def add_model_args(parser):
    parser.add_argument("--model_name", type=str, help="either the name of a codex engine, or a path to a fairseq or HF transformers serialized model. type will be inferred based on the name")
    parser.add_argument("--tokenizer_name", type=str, choices=["gpt2", "gpt2_pretokenization_newlines_only"])
    parser.add_argument("--temperature", type=float, default=0.6, help="pass 0.0 to do greedy or beam decoding")
    parser.add_argument("--top_p", type=float, default=0.95, help="nucleus top-p")
    parser.add_argument("--beam", type=int, default=1, help="beam size; only used if --temperature==0.0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_prefix")
    parser.add_argument("--candidate_scoring", choices=["mean", "sum", "random"], default="mean")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)

def add_infilling_args(parser):
    parser.add_argument("--truncation_heuristics", nargs='*', choices=TruncationParameters.HEURISTICS, default=["num_lines"])
    parser.add_argument("--bidirectional_generation", action="store_true", help="for infilling, generate candidates using both left and right contexts")
    parser.add_argument("--bidirectional_scoring", action="store_true", help="for infilling, rerank generated candidates using the left and right contexts")
    parser.add_argument("--num_candidates", type=int, default=10, help="number of candidates to use in infilling reranking")
    parser.add_argument("--max_infill_attempts", type=int, default=1, help="number of times to retry for a complete infill")

_TruncationParameters = namedtuple("_TruncationParameters", ["max_num_lines", "suffix", "is_docstring_infill", "stop_words"])
class TruncationParameters(_TruncationParameters):
    SUFFIX_NUM_CONSECUTIVE_LINES = 2

    HEURISTICS = ["num_lines", "suffix", "comment", "stop_words"]

    @staticmethod
    def from_heuristics(truncation_heuristics: List[str], missing_lines: str = None, suffix: str = None, stop_words=None):
        tp = TruncationParameters(None, None, False, None)
        for heuristic in truncation_heuristics:
            assert heuristic in TruncationParameters.HEURISTICS
            if heuristic == "num_lines":
                num_lines = len(stripped_line_split(missing_lines))
                tp = tp._replace(max_num_lines=num_lines)
            elif heuristic == "suffix":
                tp = tp._replace(suffix=suffix)
            elif heuristic == "comment":
                tp = tp._replace(is_docstring_infill=True)
            elif heuristic == "stop_words":
                tp = tp._replace(stop_words=stop_words)
            else:
                raise NotImplementedError(f"heuristic {heuristic}")
        return tp

    def truncate(self, infill: str):
        """
        Truncate an infill either to the maximum
        """
        infill_truncated = infill
        if self.suffix is not None:
            infill_truncated = truncate_overlap(infill, self.suffix, minimum_num_suffix_lines=self.SUFFIX_NUM_CONSECUTIVE_LINES)
        if self.is_docstring_infill:
            infill_truncated = truncate_docstring_infill(infill_truncated)
        if self.max_num_lines is not None:
            infill_truncated = truncate_num_lines(infill_truncated, max_num_lines=self.max_num_lines)
        if self.stop_words is not None:
            stop_index = None
            for stop_token in self.stop_words:
                if stop_token in infill_truncated:
                    index = infill_truncated.index(stop_token)
                    if stop_index is None or index < stop_index:
                        stop_index = index
            if stop_index is not None:
                infill_truncated = infill_truncated[:stop_index]
        return infill_truncated

class Model:
    def encode_stop_words(self, stop_words: List[str]):
        raise NotImplementedError()

    def complete(self, prompt: str, stop_words: List[str], **kwargs):
        text = 'DUMMY'
        choice = {
            'text': text,
            'logprobs': {
                'token_logprobs': None,
                'tokens': None,
            },
        }

        return {
            'prompt': prompt,
            'choices': [choice] * kwargs.get("n", 1),
        }

    def infill(self, parts: List[str], verbose=False, **kwargs):
        # fill in text between each string in parts
        infill = 'DUMMY'
        choice = {
            'complete': [parts[0], infill, parts[1]],
            'infills_untruncated': [infill],
            'ids': None,
            'raw': None,
            'logprobs': {
                'token_logprobs': None,
                'tokens': None,
            },
        }

        return {
            'prompt_parts': parts,
            'choices': [choice] * kwargs.get("n", 1),
        }

    def score_text(self, text_batch: List[str], scoring: str):
        # get the log probability of producing the given text autoregressively
        assert scoring in ['mean', 'sum']
        raise NotImplementedError()

    def _rank_helper(self, choices, scoring):
        assert scoring in ['mean', 'sum', 'random']
        if len(choices) == 1:
            return choices
        def scoring_fn(choice):
            if scoring == 'random':
                return random.random()
            token_logprobs = choice['logprobs']['token_logprobs']
            token_logprobs = np.array(token_logprobs)
            if scoring =='mean':
                return token_logprobs.mean()
            elif scoring == 'sum':
                return token_logprobs.sum()
            else:
                raise NotImplementedError(f"scoring {scoring}")
        return list(sorted(choices, key=scoring_fn, reverse=True))

    def rank_completions(self, prompt: str, stop_words: List[str], cached_response=None, scoring='mean', sampling=True, temperature=0.6, top_p=0.95, n=1, max_tokens=DEFAULT_MAX_TOKENS, beam=1):
        if cached_response is None:
            response = self.complete(prompt, stop_words, sampling=sampling, temperature=temperature, top_p=top_p, n=n, max_tokens=max_tokens, beam=beam)
        else:
            response = cached_response
        sorted_choices = self._rank_helper(response['choices'], scoring=scoring)
        return sorted_choices, response

    def rank_infills(self, parts: List[str], stop_words: Optional[List[str]]=None, verbose=False, bidirectional_scoring=False, bidirectional_generation=False,
                    cached_response=None, scoring='mean',
                    truncation_parameters: List[TruncationParameters] = None,
                    sampling=True, temperature=0.6, top_p=0.95, n=1, max_tokens=DEFAULT_MAX_TOKENS, beam=1):
        if truncation_parameters is None:
            truncation_parameters = [TruncationParameters(None, None, False, None) for _ in parts[:-1]]
        assert len(truncation_parameters) == len(parts) - 1

        if len(parts) != 2:
            # TODO: implement this
            raise NotImplementedError()
        else:
            prefix = parts[0]
            suffix = parts[1]
            trunc_params = truncation_parameters[0]

        if cached_response is None:
            if bidirectional_generation:
                response = self.infill(
                    [prefix, suffix],
                    stop_words=stop_words,
                    # can pass None for this since we will truncate afterward
                    truncation_parameters=None,
                    verbose=verbose,
                    sampling=sampling,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    max_tokens=max_tokens,
                    beam=beam
                )
            else:
                response = self.complete(prefix, stop_words=stop_words, sampling=sampling, temperature=temperature, top_p=top_p, n=n, max_tokens=max_tokens, beam=beam)
        else:
            response = cached_response

        choices = []
        for choice in response['choices']:
            if bidirectional_generation:
                infills_untruncated = choice['infills_untruncated']
            else:
                infills_untruncated = [choice['text']]
            assert len(infills_untruncated) == 1
            text_untruncated = infills_untruncated[0]
            text = trunc_params.truncate(text_untruncated)
            if verbose:
                print(f"--prefix:--\n{prefix}")
                print(f"--infill (truncated):--\n{text}")
                print(f"--infill (untruncated):--\n{text_untruncated}")
                print(f"--suffix:--\n{suffix}")
                if 'ids' in choice:
                    print(f"--decoded ids:--\n{self._decode(choice['ids'])}")

            def maybe_append_newline(s):
                if not s.endswith("\n"):
                    return s + "\n"
                return s
            
            d = {
                'complete': ''.join([maybe_append_newline(prefix), maybe_append_newline(text), suffix]),
                'infills': [text],
                'infills_untruncated': infills_untruncated,
                'logprobs': {
                    'token_logprobs': None,
                    'tokens': None,
                }
            }
            for key in ['infill_attempts', 'generated_all_eoss']:
                if key in choice:
                    d[key] = choice[key]
            choices.append(d)

        if bidirectional_scoring:
            completes = [choice['complete'] for choice in choices]
            scores = self.score_text(completes, scoring=scoring)
            assert len(scores) == len(choices)
            for choice, score in zip(choices, scores):
                choice['complete_score'] = score
                choice['complete_score_method'] = scoring
            sorted_choices = list(sorted(choices, key=lambda d: d['complete_score'], reverse=True))
        else:
            sorted_choices = self._rank_helper(choices, scoring)

        return sorted_choices, response

class HFModel(Model):
    def __init__(self, args, model_name, prompt_prefix=None, batch_size=None, check_low_prob_indices=None):
        super().__init__()
        self.args = args
        self.prompt_prefix = prompt_prefix
        if 'gpt-j' in model_name:
            from transformers import GPTJForCausalLM
            self.lm_model = GPTJForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
            self.lm_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        else:
            assert 'incoder' in model_name or '-hf' in model_name
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if model_name == 'facebook/incoder-6B':
                self.lm_model = AutoModelForCausalLM.from_pretrained(model_name, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
            else:
                self.lm_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.lm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm_model.eval().half().cuda()
        self.batch_size = batch_size

        self.check_low_prob_indices = check_low_prob_indices

    def _truncate_at_stop_words(self, stop_words, sequence_ids, logprobs, show_warnings=True):
        # search for stopwords, to truncate after them
        full_seq_decoded = self.lm_tokenizer.decode(sequence_ids, skip_special_tokens=False)
        min_index = None
        for stop_word in stop_words:
            index = full_seq_decoded.find(stop_word)
            if index < 0:
                continue
            if min_index is None or index < min_index:
                min_index = index
        
        if min_index is not None:
            # if you we find one of the stopwords, then we delete everything from the stopword on
            seq_decoded = full_seq_decoded[:min_index]
            # figure out how many tokens to take from log probs by reencoding the truncated string
            # TODO: this may not exactly be right since this I don't think BPE is a prefix code
            seq = self.lm_tokenizer.encode(seq_decoded, add_special_tokens=True)
            logprobs = logprobs[:len(seq)]
        else:
            if show_warnings:
                print('no stopword found!') # not having any stopword found is probably a very bad sign
            seq = sequence_ids
            seq_decoded = full_seq_decoded
        return seq, seq_decoded, logprobs

    def encode_stop_words(self, stop_words: List[str]):
            return [self.lm_tokenizer.encode(string, add_special_tokens=False) for string in stop_words]

    def complete(self, prompt, stop_words: List[str], sampling=True, max_tokens=DEFAULT_MAX_TOKENS, top_p=0.95, n=1, num_log_probs=1, temperature=0.6, beam=1):
        ''' This function runs GPT-2 locally using HF transformers but places the outputs into an json that looks just like the one
        provided by the OpenAI API. '''

        batch_size = n if self.batch_size is None else self.batch_size

        if beam != 1:
            raise NotImplementedError()

        if not sampling:
            raise NotImplementedError()

        if self.prompt_prefix is not None:
            prompt = f"{self.prompt_prefix}\n{prompt}"

        assert isinstance(prompt, str)
        prompt = [prompt] # below code assumes list

        encoded_stop_words = self.encode_stop_words(stop_words)
        # print(f"stop_words: {stop_words}")
        # print(f"encoded_stop_words: {encoded_stop_words}")

        input_ids = self.lm_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=False)
        choices = []
        lm_model = self.lm_model
        while len(choices) < n:
            num_to_sample = min(batch_size, n - len(choices))
            print(f"num_to_sample: {num_to_sample}")
            with torch.inference_mode():
                # generate from the model
                total_sequences = lm_model.generate(
                    input_ids=input_ids['input_ids'].cuda(),
                    attention_mask=input_ids['attention_mask'].cuda(),
                    max_length=max_tokens + len(input_ids['input_ids'][0]),
                    do_sample=True,
                    num_return_sequences=num_to_sample,
                    top_p=top_p,
                    early_stopping=True,
                    use_cache=True,
                    temperature=temperature,
                )

                # now do something dumb where you run the model another time to get the probs
                logits = lm_model.forward(input_ids=total_sequences, return_dict=True).logits.detach().cpu().to(torch.float32)
                # get the top tokens and probs for the generated tokens
                probs = torch.softmax(logits[:,-max_tokens-1:], dim=2).cpu()
                if self.check_low_prob_indices is not None:
                    print("checking tokens")
                    to_check = probs[...,self.check_low_prob_indices]
                    if torch.any(to_check > 1e-4):
                        print("warning: some tokens-to-check have non-zero probability")

                top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
                logprobs = torch.log(probs)
                top_log_probs = torch.log(top_probs)

            # create the return value to resemble OpenAI
            return_json = {}
            for batch_id in range(num_to_sample):
                seq = total_sequences[batch_id][-max_tokens:]
                curr_json = {}

                # package up top_log_probs and top_tokens
                curr_top_log_probs = top_log_probs[batch_id]
                curr_top_tokens = top_tokens[batch_id]
                assert len(curr_top_log_probs) == len(curr_top_tokens)
                zipped_lps = list(zip(curr_top_log_probs, curr_top_tokens))

                seq, _, zipped_lps = self._truncate_at_stop_words(stop_words, seq, zipped_lps)

                # unpack
                curr_top_log_probs, curr_top_tokens = zip(*zipped_lps)

                # cutoff the -1 here because the probs are shifted one over for LMs
                curr_top_log_probs = curr_top_log_probs[:-1]
                curr_top_tokens = curr_top_tokens[:-1]

                # fill the return json with the top tokens and probs to match the OpenAI return value.
                curr_json['logprobs'] = {}
                curr_json['logprobs']['top_logprobs'] = []
                curr_json['logprobs']['token_logprobs'] = []
                curr_json['logprobs']['tokens'] = []

                for current_element_top_log_probs, current_element_top_tokens in zip(curr_top_log_probs, curr_top_tokens):
                    # tokens is a list of the top token at each position
                    curr_json['logprobs']['tokens'].append(self.lm_tokenizer.decode([current_element_top_tokens[0]]))
                    # token_logprobs is a list of the logprob of the top token at each position
                    curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                    # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[self.lm_tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)

                curr_json['text'] = self.lm_tokenizer.decode(seq, skip_special_tokens=True)

                choices.append(curr_json)
        return_json['choices'] = choices
        return return_json

class FairseqModel(Model):
    def __init__(self, args: argparse.Namespace, model_path: str, prompt_prefix=None, batch_size=None, model=None, max_seq_length=None):
        self.args = args
        tokenizer_name = args.tokenizer_name
        if tokenizer_name is None:
            bpe = "gpt2_pretokenization_newlines_only"
        else:
            bpe = tokenizer_name
        self.bpe = bpe
        assert bpe in ["gpt2_pretokenization_newlines_only", "gpt2"], f"invalid bpe type {bpe}"
        if not model_path.endswith(".pt"):
            print(f"warning: model_path {model_path} does not end in *.pt")
        assert os.path.exists(model_path), f"model_path {model_path} should be a file"
        model_root_dir = os.path.dirname(model_path)
        model_basename = os.path.basename(model_path)
        self.gpt2_encoder_json = gpt2_encoder_json = os.path.join(model_root_dir, "vocab.json")
        self.gpt2_vocab_bpe = gpt2_vocab_bpe = os.path.join(model_root_dir, "merges.txt")

        print(f"model_root_dir: {model_root_dir}")
        print(f"model_basename: {model_basename}")
        from fairseq.models.transformer_lm import TransformerLanguageModel
        if model is None:
            self.lm_model = TransformerLanguageModel.from_pretrained(
                model_root_dir, model_basename, bpe=bpe, gpt2_encoder_json=gpt2_encoder_json, gpt2_vocab_bpe=gpt2_vocab_bpe
                ).half()
            # self.lm_model = TransformerLanguageModel.from_pretrained(
            #     model_root_dir, model_basename, bpe=bpe, gpt2_encoder_json=gpt2_encoder_json, gpt2_vocab_bpe=gpt2_vocab_bpe
            #     )
            self.lm_model.eval().cuda() 
            # self.lm_model.eval()
        else:
            self.lm_model = model

        # length normalization? 
        #self.unnormalized = args.unnormalized
        self.unnormalized = True
        self.lm_model.cfg.generation['unnormalized'] = True

        self.prompt_prefix = prompt_prefix
        self.eos_index = self.lm_model.task.dictionary.eos_index

        self.batch_size = batch_size

        self.max_seq_length = max_seq_length

    def encode_stop_words(self, stop_words: List[str]):
        # TODO: I don't think this is needed anymore
        raise NotImplementedError()
        encoded = []
        for stop_word in stop_words:
            # strip the EOS symbol
            enc = self.lm_model.encode(stop_word)
            assert enc[-1] == self.lm_model.src_dict.eos()
            encoded.append(enc[:-1].tolist())
        return encoded

    @property
    def _extra_stop_words(self):
        return ["<|", "<|/", "<code>", "</code>", "<cell>", "</cell>", "<text>", "</text>"]

    def _encode(self, text: str, strip_eos=False):
        # -> torch.tensor
        encoded = self.lm_model.encode(text)
        if strip_eos and encoded[-1].item() == self.eos_index:
            encoded = encoded[:-1]
        return encoded

    def _decode(self, tokens) -> str:
        # tokens: torch.tensor
        return self.lm_model.decode(tokens)

    def score_text(self, text_batch: List[str], scoring: str='sum'):
        tokens = [self._encode(text) for text in text_batch]
        return self.score_tokens(tokens, scoring)

    def score_tokens(self, tokens_batch, scoring='sum'):
        # tokens_batch: List[torch.tensor]
        # not sure if passing temperature here does anything, but it shouldn't hurt
        all_scores = []

        i = 0
        while len(all_scores) < len(tokens_batch):
            subbatch = []
            for seq in tokens_batch[i:i+self.batch_size]:
                if len(seq) > 2047:
                    print(f"long sample with length {len(seq)}; truncating")
                    seq = seq[:2047]
                subbatch.append(seq)
            i += self.batch_size
            ret_vals = self.lm_model.generate(subbatch, score_reference=True, temperature=1.0)
            for ret_val in ret_vals:
                assert len(ret_val) == 1
                log_probs = ret_val[0]['positional_scores']
                if scoring == 'sum':
                    score = log_probs.sum()
                elif scoring == 'mean':
                    score = log_probs.mean()
                else:
                    raise NotImplementedError(f"scoring {scoring}")
                all_scores.append(score.item())
        return all_scores

    def _generate(self, encoded_prompt: torch.tensor, max_tokens, top_p=0.95, n=1, temperature=0.6, extra_encoded_stop_words=None, all_must_complete=True):
        if isinstance(encoded_prompt, list):
            encoded_prompt = torch.tensor(encoded_prompt)
        assert encoded_prompt.dim() == 1
        if self.max_seq_length and (len(encoded_prompt) + max_tokens > self.max_seq_length):
            new_len = self.max_seq_length-(max_tokens)
            encoded_prompt = encoded_prompt[-new_len:]
        prompt_len = len(encoded_prompt)
        from priming_generator import GreedyDecoding, TopPSampling
        # strip EOS from end
        if temperature == 0:
            decoder = GreedyDecoding(self.lm_model, min_len=prompt_len, max_len=max_tokens+prompt_len, temperature=1.0, show_tqdm=False)
        else:
            decoder = TopPSampling(self.lm_model, min_len=prompt_len, max_len=max_tokens+prompt_len, sampling_topp=top_p, temperature=temperature, show_tqdm=False)
        if encoded_prompt[-1].item() == decoder.eos:
            encoded_prompt = encoded_prompt[:-1]

        decoder = decoder.to(self.lm_model.device)
        encoded_prompt = encoded_prompt.to(self.lm_model.device)

        encoded_stop_words = [[decoder.eos]]
        if extra_encoded_stop_words is not None:
            for esw in extra_encoded_stop_words:
                if isinstance(esw, torch.Tensor):
                    esw = esw.tolist()
                assert isinstance(esw, list)
                assert isinstance(esw[0], int)
                encoded_stop_words.append(esw)

        num_yielded = 0
        while num_yielded < n:
            this_batch_size = min(self.args.batch_size, n - num_yielded)
            multi_completion_tokens, multi_completion_token_log_probs, multi_completion_lengths, multi_found_stop = decoder.decode_multiple_candidates(
                encoded_prompt.to(self.lm_model.device),
                num_candidates=this_batch_size,
                encoded_stop_words=encoded_stop_words,
                all_must_complete=all_must_complete,
            )
            assert multi_completion_tokens.size(0) == multi_completion_token_log_probs.size(0) == multi_completion_lengths.size(0) == multi_found_stop.size(0) == this_batch_size
            for completion_ix in range(this_batch_size):
                completion_length = multi_completion_lengths[completion_ix].item()
                completion_tokens = multi_completion_tokens[completion_ix][:completion_length]
                completion_token_log_probs = multi_completion_token_log_probs[completion_ix][:completion_length]
                # remove initial EOS
                assert completion_tokens[0].item() == decoder.eos
                completion_tokens = completion_tokens[1:]
                completion_token_log_probs = completion_token_log_probs[1:]
                assert completion_tokens.size() == completion_token_log_probs.size()
                assert torch.allclose(completion_tokens[:len(encoded_prompt)], encoded_prompt)
                yield completion_tokens[len(encoded_prompt):], completion_token_log_probs[len(encoded_prompt):], multi_found_stop[completion_ix].item()
                num_yielded += 1

    def _truncate_at_stop_words(self, stop_words, sequence_ids, logprobs, show_warnings=True):
        # search for stopwords, to truncate after them
        full_seq_decoded = self._decode(sequence_ids)
        min_index = None
        for stop_word in stop_words:
            index = full_seq_decoded.find(stop_word)
            if index < 0:
                continue
            if min_index is None or index < min_index:
                min_index = index
        
        if min_index is not None:
            # if you we find one of the stopwords, then we delete everything from the stopword on
            seq_decoded = full_seq_decoded[:min_index]
            # figure out how many tokens to take from log probs by reencoding the truncated string
            # TODO: this may not exactly be right since this I don't think BPE is a prefix code
            seq = self._encode(seq_decoded, strip_eos=True)
            logprobs = logprobs[:len(seq)]
        else:
            if show_warnings:
                print('no stopword found!') # not having any stopword found is probably a very bad sign
            seq = sequence_ids
            seq_decoded = full_seq_decoded
        return seq, seq_decoded, logprobs

    def complete(self, prompt: str, stop_words: List[str], sampling=True, max_tokens=DEFAULT_MAX_TOKENS, top_p=0.95, n=1, num_log_probs=1, temperature=0.6, beam=1):
        ''' This function runs fairseq LM locally but places the outputs into an json that looks just like the one
        provided by the OpenAI API. '''

        assert beam == 1, "beam search is not implemented"

        batch_size = n if self.batch_size is None else self.batch_size

        if not sampling:
            raise NotImplementedError()

        if num_log_probs != 1:
            raise NotImplementedError()

        stop_words = stop_words + self._extra_stop_words

        if self.prompt_prefix is not None:
            # TODO: add option to not insert newline
            prompt = f"{self.prompt_prefix}\n{prompt}"

        encoded_prompt = self._encode(prompt, strip_eos=True)
        encoded_stop_words = [self._encode(stop_word, strip_eos=True).tolist() for stop_word in stop_words]

        all_tokens = []
        all_log_probs = []
        for tokens, log_probs, found_stop in self._generate(
            encoded_prompt,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            temperature=temperature,
            extra_encoded_stop_words=encoded_stop_words,
        ):
            all_tokens.append(tokens)
            all_log_probs.append(log_probs)

        assert len(all_tokens) == n
        assert len(all_log_probs) == n

        # construct batched tokens
        # create the return value to resemble OpenAI
        choices = []
        return_json = {}
        for completion_ix in range(len(all_tokens)):
            curr_json = {}

            full_seq = all_tokens[completion_ix].cpu()
            full_logprobs = all_log_probs[completion_ix]
            assert len(full_seq) == len(full_logprobs)

            seq, seq_decoded, logprobs = self._truncate_at_stop_words(stop_words, full_seq, full_logprobs, show_warnings=True)

            curr_json['text'] = seq_decoded
        
            # fill the return json with the top tokens and probs to match the OpenAI return value.
            curr_json['logprobs'] = {}
            curr_json['logprobs']['token_logprobs'] = logprobs.tolist() 
            curr_json['logprobs']['tokens'] = [self._decode([ix]) for ix in seq]

            # TODO: add top_logprobs
            # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}

            choices.append(curr_json)

        # {'choices': [{'text': text, 'logprobs': ... }]}
        return_json['choices'] = choices
        return return_json

class CausalMasking(FairseqModel):
    EOSS = "<eoss>"

    TOKENIZER_OFFSET = 4

    @staticmethod
    def make_sentinel(i):
        return f"<sentinel:{i}>"

    def sentinel_id(self, i):
        return self.tokenizer.token_to_id(self.make_sentinel(i)) + self.TOKENIZER_OFFSET

    @property
    def _sentinel_tokens(self):
        return [CausalMasking.make_sentinel(i) for i in range(256)]

    @property
    def _special_tokens(self):
        return self._sentinel_tokens + [self.EOSS]

    @property
    def _extra_stop_words(self):
        return super()._extra_stop_words + self._special_tokens

    def __init__(self, args, model_path: str, prompt_prefix=None, batch_size=None, model=None):
        super().__init__(args, model_path, prompt_prefix, batch_size=batch_size, model=model)
        from tokenizers import ByteLevelBPETokenizer
        self.tokenizer = tokenizer = ByteLevelBPETokenizer.from_file(
            # these will be set by super().__init__
            self.gpt2_encoder_json, self.gpt2_vocab_bpe,
            pretokenizer_split_newlines_only=(self.bpe=="gpt2_pretokenization_newlines_only"),
        )
        tokenizer.add_special_tokens(self._special_tokens)

        self.EOSS_ID = tokenizer.token_to_id(self.EOSS) + self.TOKENIZER_OFFSET

        self.extra_sentinel = True

    def _encode(self, text, strip_eos=False):
        if not strip_eos:
            return torch.tensor(self.tokenizer.encode(text).ids + [self.eos_index - self.TOKENIZER_OFFSET]) + self.TOKENIZER_OFFSET
        else:
            return torch.tensor(self.tokenizer.encode(text).ids) + self.TOKENIZER_OFFSET

    def _decode(self, tokens):
        token_ids = torch.tensor(tokens)
        token_ids_offset = token_ids - self.TOKENIZER_OFFSET
        # for i in range(len(token_ids_offset)):
        #     if token_ids_offset[i] < 0:
        #         print(f"warning: found invalid token {token_ids_offset[i]} at index {i} in {token_ids_offset}")
        #         token_ids_offset = token_ids_offset[:i]
        #         break
        return self.tokenizer.decode((token_ids_offset).tolist(), skip_special_tokens=False)

    def infill(self, parts: List[str], stop_words: Optional[List[str]]=None, verbose=False, n=1, truncation_parameters: List[TruncationParameters]=None, sampling=True, max_tokens=DEFAULT_MAX_TOKENS, top_p=0.95, temperature=0.0, beam=1):
        # Force the model to fill in code in between each string in parts
        # see code_to_docstring and docstring_to_code for example usages
        if truncation_parameters is None:
            truncation_parameters = [TruncationParameters(None, None, False, None) for _ in parts[:-1]]
        else:
            raise NotImplementedError("truncation_parameters for infill()")
        
        assert len(truncation_parameters) == len(parts) - 1
        model = self.lm_model
        assert isinstance(parts, list)
        assert len(parts) > 1

        if not sampling:
            raise NotImplementedError()
        
        if n != 1:
            raise NotImplementedError()

        if beam != 1:
            raise NotImplementedError()

        parts_without_prompt_prefix = parts
        if self.prompt_prefix is not None:
            parts = parts.copy()
            parts[0] = f"{self.prompt_prefix}\n{parts[0]}"

        infills = []

        prompt = []

        if stop_words is None:
            stop_words = []

        stop_words = stop_words + self._extra_stop_words
        encoded_stop_words = [self._encode(stop_word, strip_eos=True).tolist() for stop_word in stop_words]
        encoded_stop_words.append([self.EOSS_ID])

        # encode parts separated by sentinel
        for sentinel_ix, part in enumerate(parts):
            part_tokens = self._encode(part, strip_eos=True)
            prompt.extend(part_tokens.tolist())
            if sentinel_ix < len(parts) - 1:
                prompt.append(self.sentinel_id(sentinel_ix))
            else:
                # only makes sense to add an extra sentinel if we do have some text coming later, otherwise, we tend to just end the region immediately
                if self.extra_sentinel and len(part) > 0:
                    prompt.append(self.sentinel_id(sentinel_ix))

        ids = prompt.copy()
        infills = []

        complete = []

        infill_scores = []
        infill_tokens = []

        generated_all_eoss = []

        attempt_nums = []

        # autoregressively fill in
        for sentinel_ix, part in enumerate(parts[:-1]):
            ids.append(self.sentinel_id(sentinel_ix))
            model.cfg.generation['max_len_b'] = max_tokens + len(ids)
            if verbose:
                print(part, end="")
                print(f"<sentinel:{sentinel_ix}>", end="")
            with torch.no_grad():
                # print("completing: ")
                # print(self._decode(ids))
                attempt_num = 0
                generated_eoss = False
                for completion, scores, found_stop in self._generate(
                    torch.tensor(ids),
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=self.args.max_infill_attempts,
                    temperature=temperature,
                    extra_encoded_stop_words=encoded_stop_words,
                    all_must_complete=True,
                ):
                    attempt_num += 1
                    completion = completion.tolist()
                    # allow ourselves up to max_infill_attempts to find a stop word:
                    # TODO: found_stop is True if self.EOSS_ID in completion, since we passed EOSS as a stopword (verify this; and roll in)
                    if self.EOSS_ID in completion or found_stop:
                        generated_eoss = True
                        break
                if completion[-1] == 2:
                    completion = completion[:-1]
                    scores = scores[:-1]

            attempt_nums.append(attempt_num)
            generated_all_eoss.append(generated_eoss)

            # TODO: make sure to check that EOS are accounted for in the prefix removal below 
            # (they may have been added, so len(ids) might not be the right thing to use? maybe 
            # other stuff too)

            if self.EOSS_ID in completion:
                t = completion.index(self.EOSS_ID)+1
                completion = completion[:t]
                scores = scores[:t]
                # TODO: handle this better: we do want to include the score for EOSS (if we use these scores somewhere)
                # but how do we handle the case where EOSS is not present (below) without biasing toward those candidates?
                scores_no_eoss = scores[:-1]
            else:
                if verbose:
                    print(f"warning: {self.EOSS} not found; completion len {len(completion)}", file=sys.stderr)
                    print("last part:")
                    print(part)
                    print("next part:")
                    print(parts[sentinel_ix+1])
                    print("completion")
                    print(completion)
                    print(self._decode(completion))
                scores_no_eoss = scores
                completion = completion + [self.EOSS_ID]
                generated_all_eoss = False
            ids.extend(completion)

            completion_no_eoss = completion[:-1]

            completion_no_stopwords, completion_no_stopwords_decoded, scores_no_stopwords = self._truncate_at_stop_words(stop_words, completion_no_eoss, scores_no_eoss, show_warnings=False)

            infill_tokens.append(completion_no_stopwords)
            infill_scores.append(scores_no_stopwords)

            complete.append(parts_without_prompt_prefix[sentinel_ix])
            complete.append(completion_no_stopwords_decoded)
            infills.append(completion_no_stopwords_decoded)

        complete.append(parts[-1])

        choice = {
            'complete': complete,
            'infills_untruncated': infills,
            'ids': ids,
            'infill_attempts': attempt_nums,
            'generated_all_eoss': generated_all_eoss,
            # 'raw': self._decode(ids),
            'logprobs': {
                'token_logprobs': infill_scores,
                'tokens': None,
            },
        }

        return {
            'prompt_parts': parts,
            'choices': [choice],
        }

class OpenAIModel(Model):
    END_OF_TEXT = '<|endoftext|>'

    def __init__(self, args, engine='davinci-codex', persistent=True, prompt_prefix=None):
        self.args = args
        if prompt_prefix is not None:
            raise NotImplementedError("--prompt_prefix for OpenAIModel")
        self.engine = engine
        self.persistent = persistent
        # turn off the logging, which prints an HTTP request code for every call to the API 
        import logging
        logging.disable(logging.INFO)
        #logging.basicConfig(level=logging.WARNING)

    def encode_stop_words(self, stop_words: List[str]):
        return stop_words

    def score_text(self, text_batch: List[str], scoring: str):
        all_scores = []
        if scoring == 'random':
            return [random.random() for _ in text_batch]
        for text in text_batch:
            response = self.complete(text, None, max_tokens=0, temperature=1.0, echo=True)
            choice = response['choices'][0]
            token_log_probs = choice['logprobs']['token_logprobs']
            tokens = choice['logprobs']['tokens']

            if self.END_OF_TEXT in tokens:
                ix = tokens.index(self.END_OF_TEXT)
                # keep the token, so that we have the score for ending the seq, but remove everything afterward
                tokens = tokens[:ix+1]
                token_log_probs = token_log_probs[:ix+1]

            # remove None at the beginning -- no probability for the initial token, since there's no BOS
            token_log_probs = token_log_probs[1:]
            if scoring == 'sum':
                score = np.sum(token_log_probs)
            elif scoring == 'mean':
                score = np.mean(token_log_probs)
            else:
                raise NotImplementedError(f"scoring {scoring}")
            all_scores.append(score)
        return all_scores

    def _call(self, **kwargs):
        import openai
        from secret import API_KEY
        openai.api_key = API_KEY

        succeeded = False
        tries = 0
        while not succeeded:
            tries += 1
            if tries > CODEX_MAX_RETRIES:
                raise Exception("max number of retries failed")
            try:
                response = openai.Completion.create(
                    **kwargs
                )
                succeeded = True
            except (openai.error.RateLimitError, openai.error.APIError) as e:
                if not self.persistent:
                    raise e
                print(e)
                time.sleep(CODEX_RETRY_DELAY_SECONDS)
        return response

    def complete(self, prompt, stop_words, max_tokens=450, top_p=0.95, temperature=0.6, sampling=True, beam=1, n=1, **kwargs):
        if stop_words == []:
            stop_words = None
        if not sampling:
            raise NotImplementedError()
        if beam != 1:
            raise NotImplementedError()
        return self._call(
            engine=self.engine,
            prompt=prompt,
            stop=stop_words[:4],
            logprobs=1,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            n=n,
            **kwargs
        )

    def infill(self, parts: List[str], stop_words:Optional[List[str]]=None, verbose=False, n=1, truncation_parameters: List[TruncationParameters]=None, sampling=True, max_tokens=DEFAULT_MAX_TOKENS, top_p=0.95, temperature=0.0, beam=1, **kwargs):
        # inclusive_stop_words: include these in the returned value
        stop_words = None
        if not sampling:
            raise NotImplementedError()
        if beam != 1:
            raise NotImplementedError()

        if len(parts) != 2:
            raise NotImplementedError("can't infill more than 2 parts")
        if truncation_parameters is None:
            trunc_params = TruncationParameters(None, None, False, None)
        else:
            assert len(truncation_parameters) == 1
            trunc_params = truncation_parameters[0]
        choices = []
        for attempt_num in range(1, self.args.max_infill_attempts+1):
            response = self._call(
                engine=self.engine,
                prompt=parts[0],
                suffix=parts[1],
                stop=stop_words[:4],
                logprobs=1,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                n=n,
                **kwargs,
            )
            finished_choices = [choice for choice in response['choices'] if choice['finish_reason'] == 'stop']
            unfinished_choices = [choice for choice in response['choices'] if choice['finish_reason'] != 'stop']
            # prioritize finished completions
            choices.extend(finished_choices)
            if len(choices) >= n:
                break
        # but if necessary, use unfinished choices
        if len(choices) < n:
            choices.extend(unfinished_choices)
        choices = choices[:n]
        assert len(choices) == n
        for choice in choices:
            choice['infills_untruncated'] = [choice['text']]
            truncated = trunc_params.truncate(choice['text'])
            # api doesn't let us set this to an empty string
            if len(truncated) > 0:
                choice['text'] = truncated
        response['choices'] = choices
        return response

def make_model(args, cached_model=None):
    model_name = args.model_name
    print(f"guessing model type from {model_name}")
    if model_name is None:
        return Model()
    tokenizer_name = args.tokenizer_name
    prompt_prefix = args.prompt_prefix
    if 'davinci' in model_name or 'cushman' in model_name:
        if prompt_prefix is not None:
            raise NotImplementedError("prompt prefix for codex models")
        return OpenAIModel(args, model_name, persistent=True)
    elif 'incoder' in model_name or '-hf' in model_name:
        return HFModel(args, model_name, prompt_prefix=prompt_prefix, batch_size=args.batch_size)
    elif 'fairseq' in model_name or '/checkpoint' in model_name:
        if "gpt2tok" in model_name:
            assert tokenizer_name == "gpt2"
        else:
            assert tokenizer_name is None or tokenizer_name == "gpt2_pretokenization_newlines_only"
        if 'cm-' in model_name:
            return CausalMasking(args, model_name, prompt_prefix=prompt_prefix, batch_size=args.batch_size, model=cached_model)
        elif 'lm-' in model_name:
            return FairseqModel(args, model_name, prompt_prefix=prompt_prefix, batch_size=args.batch_size, model=cached_model)
        elif 'gpt-j' in model_name:
            if prompt_prefix is not None:
                raise NotImplementedError()
            return HFModel(args, model_name, tokenizer_name, batch_size=args.batch_size)
        else:
            raise ValueError(f"couldn't guess model type from {model_name}")
    elif model_name == 'code-gpt2':
        if prompt_prefix is not None:
            raise NotImplementedError()
        return CodeGPT2()
    else:
        if prompt_prefix is not None:
            raise NotImplementedError()
        return HFModel(args, model_name, tokenizer_name, batch_size=args.batch_size)
