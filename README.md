# InCoder: A Generative Model for Code Infilling and Synthesis

Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi, Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, Mike Lewis

This repository hosts example code showing how to use the model using HuggingFace's `transformers` library.

See [our project site](https://sites.google.com/view/incoder-code-models) for more information, or [our paper](paper/InCoder-4-12-22.pdf), or [examples](https://sites.google.com/view/incoder-code-models/home/examples).

Code to replicate the evaluation results in our paper (in Fairseq, which we used to train the model) is coming soon!

## Requirements

`pytorch`, `tokenizers`, and `transformers``.
Our model requires HF's tokenizers >= 0.12.1, due to changes in the pretokenizer. This version is close to release, but in the meantime you can install directly from source via pip.

```
pip install pytorch
pip install git+https://github.com/huggingface/tokenizers
pip install git+https://github.com/huggingface/transformers
```

## Usage

See [example_usage.py](example_usage.py) for a demo script showing how to use the infilling capability of the model. Set BIG_MODEL = True in the script to use the 6.7B parameter model; the 1.3B will be used otherwise.

## Paper

See [our paper](paper/InCoder-4-12-22.pdf) for research details on the method, training data, models, and experimental results.


## Demo

See a demo of the 6.7B model on [HF Spaces](https://huggingface.co/spaces/facebook/incoder-demo).

## License

CC-BY-NC 4.0

## Credits

Thanks to Lucile Saulnier, Leandro von Werra, Nicolas Patry, Suraj Patil, Omar
Sanseviero, and others at HuggingFace for help with the model release, and to
Naman Goyal and Stephen Roller for the code our demo was based on!
