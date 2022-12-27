# InCoder: A Generative Model for Code Infilling and Synthesis

Daniel Fried*, Armen Aghajanyan*, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi, Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, and Mike Lewis

This repository hosts example code showing how to use the model using HuggingFace's `transformers` library. Code to replicate the evaluation results in our paper (in Fairseq, which we used to train the model) is coming soon!

See [our project site](https://sites.google.com/view/incoder-code-models) for more information, or [our paper](https://arxiv.org/abs/2204.05999), or [examples](https://sites.google.com/view/incoder-code-models/home/examples).


## Models

You can obtain the models from HuggingFace's hub:

- 6.7B parameter model: [facebook/incoder-6B](https://huggingface.co/facebook/incoder-6B)
- 1.3B parameter model: [facebook/incoder-1B](https://huggingface.co/facebook/incoder-1B)

## Tokenizer

We use a custom tokenizer, which you can load from either "facebook/incoder-1B" or "facebook/incoder-6B" (they are identical). The model was trained with padding on the left-hand side of inputs, using a `<pad>` token which has ID 1.

```
tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B") # or "facebook/incoder-6B"
tokenizer.pad_token = "<pad>"
tokenizer.padding_side = "left"
```

When calling `tokenizer.decode`, it's important to pass `clean_up_tokenization_spaces=False` to avoid removing spaces after punctuation. For example:

`tokenizer.decode(tokenizer.encode("from ."), clean_up_tokenization_spaces=False)`

(Note: encoding prepends the `<|endoftext|>` token, as this marks the start of a document to our model. This token can be removed from the decoded output by passing `skip_special_tokens=True` to `tokenizer.decode`.)

## Requirements

`pytorch`, `tokenizers`, and `transformers`.
Our model requires HF's tokenizers >= 0.12.1, due to changes in the pretokenizer.

```
pip install torch
pip install 'tokenizers>=0.12'
pip install transformers
```

## Usage

See [example_usage.py](example_usage.py) for a demo script showing how to use the infilling capability of the model. Set BIG_MODEL = True in the script to use the 6.7B parameter model; the 1.3B will be used otherwise.

For an example of batched generation, see [example_batched_usage.py](example_batched_usage.py).


## Paper

See [our paper](https://arxiv.org/abs/2204.05999) for research details on the method, training data, models, and experimental results.


## Demo

See a demo of the 6.7B model on [HF Spaces](https://huggingface.co/spaces/facebook/incoder-demo).

## License

CC-BY-NC 4.0

## Credits

Thanks to Lucile Saulnier, Leandro von Werra, Nicolas Patry, Suraj Patil, Omar
Sanseviero, and others at HuggingFace for help with the model release, and to
Naman Goyal and Stephen Roller for the code our demo was based on!
