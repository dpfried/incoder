# modified from code by Freda Shi
import argparse
import os
import random
import json
import tempfile
import torch
import os
import signal
import subprocess
import threading
import functools
from tqdm import tqdm
from functools import partial
import sys
import pprint

from models import make_model, add_model_args
from utils import dump_git_status, dump_version_info

# stop words differ from human_eval because we don't have the function signature
# TODO: consider standardizing these interfaces
MBPP_STOP_WORDS = ["\nassert", "\nclass", "\nif", '\n"""', "\nprint", "[DONE]"]

class MBPPDataset(object):
    def __init__(self, path='/private/home/fhs/data/mbpp/mbpp.jsonl'):
        self.data = [json.loads(line) for line in open(path)]
        split_ids = self.split_ids = {
            'evaluation': list(range(11, 510+1)),
            # 'prompting': list(range(1, 10+1)),
            # Jacob Austin said that they use 2,3,4 for few-shot experiments, so start with 2
            'prompting': list(range(2, 10+1)) + [1],
            'training': list(range(511, 1000+1)),
        }
        for k1 in split_ids:
            for k2 in split_ids:
                if k1 == k2:
                    continue
                assert len(set(split_ids[k1]) & set(split_ids[k2])) == 0, f"overlap between {k1} and {k2}"
        # use a subset of training for validation
        split_ids['validation'] = split_ids['training'][::4]
        def filt(split_name):
            return [d for d in self.data if d['task_id'] in split_ids[split_name]]
        self.data_splits = {
            split_name: filt(split_name)
            for split_name in split_ids.keys()
        }
        assert len(self.data_splits['evaluation']) == 500

class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True, preexec_fn=os.setsid)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            os.killpg(self.process.pid, signal.SIGTERM)
            thread.join()
        return self.process.returncode

def comment_prompt(instance, include_solution=False):
    description = instance['text']
    test_example = instance['test_list'][0]
    prompt = f'"""\n{description}\n{test_example}\n"""\n'

    if include_solution:
        prompt += f"{instance['code']}\n"
    return prompt

def google_prompt(instance, include_solution=False, include_delimiters=False):
    description = instance['text']
    test_example = instance['test_list'][0]
    test_list = '\n'.join(instance['test_list'])
    prompt = f"You are an expert Python programmer, and here is your task: {description} Your code should pass these tests:\n\n{test_list}\n"

    if include_delimiters:
        prompt += '[BEGIN]\n'

    if include_solution:
        prompt += f"{instance['code']}\n"
        if include_delimiters:
            prompt += '[DONE]\n'
    return prompt

def evaluate_code_generic(args, model):
    """
    prompt_completion_function should be prompt function of the sort: lambda prompt: str -> {
        ...,
        'choices': [{
            ...,
            'text': completion: str,
            ...,
        }],
        ...
    }
    """
    num_candidates_generated = args.num_candidates_generated
    num_candidates_evaluated = args.num_candidates_evaluated
    output_path = args.output_path

    tempdir = tempfile.TemporaryDirectory(prefix='mbpp-eval')
    # this will be used temporarily for all solutions
    code_path = f'{tempdir.name}/code.py'
    problems_passed = 0
    total_problems = 0
    successes = 0
    attempts = 0

    has_k_shot = args.k_shot is not None and args.k_shot > 0

    prompt_function = {
        "google": functools.partial(google_prompt, include_delimiters=has_k_shot),
        "comment": comment_prompt,
    }[args.prompt_template]

    if has_k_shot:
        prompt_prefix = '\n'.join(prompt_function(item, include_solution=True) 
                                  for item in dataset.data_splits['prompting'][:args.k_shot])
    else:
        prompt_prefix = ''

    try:
        os.system(f'mkdir -p {output_path}')
        print(f"split: {args.split}")
        data = dataset.data_splits[args.split]
        if args.n_examples:
            print(f"filtering to {args.n_examples} examples")
            data = data[:args.n_examples]
        bar = tqdm(data, ncols=80)
        for i, item in enumerate(bar):
            this_attempts = 0
            some_attempt_passed = False
            text = item['text']
            test_setups = item['test_setup_code']
            test_cases = item['test_list']
            prompt = f'{prompt_prefix}{prompt_function(item, include_solution=False)}'

            if num_candidates_evaluated != num_candidates_generated:
                raise NotImplementedError()
            response = model.complete(
                prompt, MBPP_STOP_WORDS, n=num_candidates_generated, max_tokens=2048 if has_k_shot else args.max_tokens,
                temperature=args.temperature, top_p=args.top_p,
            )
            all_code = [choice['text'] for choice in response['choices']]
            for code_ix, code in enumerate(all_code):
                verbose = args.verbose and (args.verbose_candidates is None or code_ix < args.verbose_candidates)
                this_attempts += 1
                if verbose:
                    print(f"PROBLEM {i} ATTEMPT {this_attempts}")
                solution_path = f'{output_path}/solution-{i:03d}-{this_attempts}.py'
                # if os.path.exists(solution_path):  # fix bug: if prompts don't match, rerun
                #     existing_prompt = ''.join(open(solution_path).readlines()[:5]).strip()
                #     if existing_prompt == prompt:
                #         continue

                if verbose:
                    print("<PROMPT>")
                    print(prompt)
                    print("</PROMPT>")

                # write code to file 
                with open(code_path, 'w') as fout:
                    #print(prompt, file=fout)
                    print(code, file=fout)
                    if verbose:
                        print("<COMPLETION>")
                        print(code)
                        print("</COMPLETION>")
                    print(test_setups, file=fout)
                    if verbose:
                        print("<TESTS>")
                        print(test_setups)
                    for case in test_cases:
                        print(case, file=fout)
                        if verbose:
                            print(case)
                    if verbose:
                        print("</TESTS>")
                    fout.close()
                os.system(f'cp {code_path} {solution_path}')
                command = Command(f'python {code_path} >/dev/null 2>&1')
                this_passed = (command.run(timeout=args.timeout) == 0)
                successes += this_passed
                if verbose:
                    print("PASSED" if this_passed else "FAILED")
                    print()
                attempts += 1
                some_attempt_passed |= this_passed
                #bar.set_description(f'{successes}/{attempts} successes.')
            if some_attempt_passed:
                problems_passed += 1
            total_problems += 1
            bar.set_description(f'ps: {problems_passed / total_problems * 100:.1f}%; as: {successes/attempts*100:.1f}%')
            if args.verbose:
                print(f'ps: {problems_passed / total_problems * 100:.1f}%; as: {successes/attempts*100:.1f}%')
        tempdir.cleanup()
    except KeyboardInterrupt:
        tempdir.cleanup()
    print(f'problem success: {problems_passed}/{total_problems} passed ({problems_passed/total_problems*100:.2f}%): ')
    print(f'attempt success: {successes}/{attempts} ({successes/attempts*100:.2f}%); ({attempts / total_problems:.2f} per-problem)')
    return problems_passed / total_problems

def make_parser():
    parser = argparse.ArgumentParser()
    add_model_args(parser)

    parser.add_argument('--timeout', type=int, default=10)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument("--num_candidates_generated", type=int, default=15)
    parser.add_argument("--num_candidates_evaluated", type=int, default=15)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--verbose_candidates', type=int, default=5)

    parser.add_argument("--k_shot", type=int)
    parser.add_argument("--prompt_template", choices=["google", "comment"], default="comment")
    parser.add_argument("--split", choices=["evaluation", "prompting", "training", "validation"], default="evaluation")
    parser.add_argument("--n_examples", type=int)

    parser.add_argument("--git_status", action="store_true")
    return parser

if __name__ == '__main__':
    print(' '.join(sys.argv))
    parser = make_parser()
    args = parser.parse_args()

    pprint.pprint(vars(args))
    if args.git_status:
        dump_git_status()
        dump_version_info()

    random.seed(1)

    dataset = MBPPDataset()

    model = make_model(args)
    evaluate_code_generic(args, model)
