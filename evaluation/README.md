# InCoder Evaluation

This is a work in progress! Full code coming soon, after we finish integrating the HF model into our evaluation harness.

It currently contains the code for left-to-right evaluation on HumanEval (Chen et al.)

## Requirements

`torch, tokenizers>=0.12, transformers`, as for the HF model (see [README.md](../README.md)).

In addition, you'll need [my fork of HumanEval](https://github.com/dpfried/human-eval).

## Evaluation

### HumanEval left-to-right generation

To evaluate the 6B model, with 20 candidates generated, and a temperature of 0.2, run

1) `./he_hf-incoder.sh facebook/incoder-6B 20 0.2 trial1 --verbose  --prompt_prefix "<| file ext=.py|>"`

Scores reported will be pass@1 at the conclusion of a single randomly chosen candidate from the 20. To perform the unbiased estimation from the Chen et al. paper using these 20 candidates, run

2) `python he.py --num_candidates_evaluated 20 --cached_responses --candidate_scoring random --multiple_cached_responses_filenames expts/he/incoder-6B_last_pg_ncg-20_temp-0.2/trial1/responses.pkl`

To get the numbers reported in the paper for pass@1, we used 200 candidates (following Chen et al.). To do this, e.g. use 10 separate parallel runs of step (1) (trial1... trial10) and then run

`python he.py --num_candidates_evaluated 100 --cached_responses --candidate_scoring random --multiple_cached_responses_filenames expts/he/incoder-6B_last_pg_ncg-20_temp-0.2/trial*/responses.pkl`

pass@10 and pass@100 scores use temperature 0.8 (following Chen et al.'s, use of different temperatures for each pass@k metric.)
