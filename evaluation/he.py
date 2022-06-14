import sys
import json
import time
import tqdm
import pickle
import pprint
import argparse
import numpy as np
from collections import defaultdict

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

from utils import build_systematic_infill_prompt, all_equal, unpickle, dump_git_status, dump_version_info

from models import make_model, Model, add_model_args

# can't include print since the codex API will only handle up to 4 stop words
HUMAN_EVAL_STOP_WORDS = ["\nclass", "\ndef", "\n#", "\nif"]

def combine_responses(list_of_responses):
    assert all_equal(resp.keys() for resp in list_of_responses)
    responses = {}
    for resp in list_of_responses:
        for problem_id in resp:
            if problem_id not in responses:
                responses[problem_id] = {'choices': []}
            responses[problem_id]['choices'].extend(resp[problem_id]['choices'])
    return responses

def generate_he_infill_problems(args, eval_type="one_line"):
    """Masks out a subset of lines in the HumanEval reference solution."""
    assert eval_type in ("one_line", "all_lines")
    problems = list(sorted(read_problems().items()))

    for i, (task_id, problem) in enumerate(problems):
        soln = problem["canonical_solution"].rstrip() # note we strip extra newlines
        lines = soln.split("\n")
        num_lines = len(lines)
        
        num_lines_to_mask = []        
        if eval_type == "one_line":
            for num_before in range(0, num_lines):
                num_lines_to_mask.append((num_before, num_lines - num_before - 1))
        else:
            for num_before in range(0, num_lines):
                for num_after in range(0, num_lines - num_before):
                    num_lines_to_mask.append((num_before, num_after))

        task_id_problems = []

        for num_before, num_after in num_lines_to_mask:
            prompt_parts, missing_lines = build_systematic_infill_prompt(
                    problem["prompt"],
                    soln,
                    num_before,
                    num_after)

            # if this region is all whitespace, skip it
            if not missing_lines.strip():
                continue
            
            task_id_problems.append({
                "task_id": task_id,
                "num_before": num_before,
                "num_after": num_after,
                "missing_lines": missing_lines, 
                "prompt_parts": prompt_parts,
                "canonical_solution": problem["canonical_solution"],
            })
        yield task_id, task_id_problems

def make_parser():
    parser = argparse.ArgumentParser()

    add_model_args(parser)

    parser.add_argument("--num_problems", type=int)
    parser.add_argument("--num_candidates_generated", type=int, default=15)
    parser.add_argument("--num_candidates_evaluated", type=int, default=1)
    parser.add_argument("--output_filename", default="samples.jsonl")
    parser.add_argument("--response_filename", default="responses.pkl")
    parser.add_argument("--cached_responses", action='store_true')
    parser.add_argument("--multiple_cached_responses_filenames", nargs='*')
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--git_status", action="store_true")

    return parser


if __name__ == "__main__":
    print(' '.join(sys.argv))

    parser = make_parser()
    args = parser.parse_args()

    pprint.pprint(vars(args))
    if args.git_status:
        dump_git_status()
        dump_version_info()

    model = make_model(args)

    problems = list(sorted(read_problems().items()))
    if args.num_problems is not None:
        problems = problems[:args.num_problems]

    samples_to_evaluate = []
    if args.cached_responses:
        if args.multiple_cached_responses_filenames:
            responses = combine_responses([unpickle(fname) for fname in args.multiple_cached_responses_filenames])
        else:
            responses = unpickle(args.response_filename)
    else:
        responses = {}

    all_results = []
    all_extras = []

    with tqdm.tqdm(problems, ncols=120) as pbar:
        for task_id, problem in pbar:
            prompt = problem['prompt']
            # candidates: [{'text': text, 'logprobs': {...}}, ...]
            candidates, response = model.rank_completions(
                prompt, HUMAN_EVAL_STOP_WORDS,
                max_tokens=args.max_tokens,
                n=args.num_candidates_generated,
                # if we've cached responses, use the cached
                cached_response=responses.get(task_id) if args.cached_responses else None,
                scoring=args.candidate_scoring,
                temperature=args.temperature,
                top_p=args.top_p,
                beam=args.beam,
            )
            responses[task_id] = response
            this_samples_to_evaluate = []
            for candidate in candidates[:args.num_candidates_evaluated]:
                if args.verbose:
                    print("prompt:")
                    print(prompt)
                    print("candidate:")
                    print(candidate["text"])
                    print("canonical solution:")
                    print(problem["canonical_solution"])
                    print()
                this_samples_to_evaluate.append(dict(
                    task_id=task_id,
                    completion=candidate["text"]
                ))
            samples_to_evaluate.extend(this_samples_to_evaluate)

            this_results, this_extra = evaluate_functional_correctness(sample_file=None, samples=this_samples_to_evaluate, suppress=True, strict=False)
            all_results.append(this_results)
            all_extras.append(this_extra)
            average_pass_at_1 = np.mean([res['pass@1'] for res in all_results])
            pbar.set_postfix({'pass@1': average_pass_at_1})

    write_jsonl(args.output_filename, samples_to_evaluate)
    with open(args.response_filename, 'wb') as f:
        pickle.dump(responses, f)

    import pprint
    results, extra = evaluate_functional_correctness(sample_file=None, samples=samples_to_evaluate)
    pprint.pprint(results)
