from typing import List, Tuple
import pickle
import json
import sys
import subprocess
import re

TRIPLE_QUOTE = '"""'
SINGLE_TRIPLE_QUOTE = "'''"
SPACES4 = " " * 4
SPACES8 = " " * 8
EOF = "<|/ file |>"

def standardize_docstring_prompt(prefix: str, suffix: str) -> str:
    """Strips any existing docstring delimiters from the prompt prefix and suffix
    and adds our own delimiter and whitespace.

    Note lots of edge cases being handled here:
    - codexglue docstring text sometimes contains the docstring delimiters, inconsistently
    - suffix can contain other functions with docstrings
    - prefix should keep the correct indentation for the whitespace
    """
    original_delim = None

    for delim in [TRIPLE_QUOTE, SINGLE_TRIPLE_QUOTE]:
        if delim in prefix:
            prefix = prefix[:prefix.index(delim)]
            original_delim = delim
            break

    # Need to be more careful about looking for single quote delimiters,
    #  since they can be used in strings
    single_single_quote_with_trailing_spaces = re.compile(r'[^\'"][\']\s*$')
    if single_single_quote_with_trailing_spaces.search(prefix):
        prefix = prefix[:single_single_quote_with_trailing_spaces.search(prefix).start()]
        original_delim = "'"

    single_double_quote_with_trailing_spaces = re.compile(r'[^\'"]["]\s*$')
    if single_double_quote_with_trailing_spaces.search(prefix):
        prefix = prefix[:single_double_quote_with_trailing_spaces.search(prefix).start()]
        original_delim = '"'

    # If we know the original delimiter, we can remove it from the suffix
    if original_delim is not None:
        if original_delim in suffix:
            suffix = suffix[suffix.index(original_delim) + len(original_delim):]
    # Delimiter not in prefix, check we don't have a delimiter in suffix
    else:
        triple_quote_with_leading_spaces = re.compile(r'^\s*(\'\'\'|""")')
        if triple_quote_with_leading_spaces.search(suffix):
            suffix = suffix[triple_quote_with_leading_spaces.search(suffix).end():]

        single_quote_with_leading_spaces = re.compile(r'^\s*[\'"]\s*\n')
        if single_quote_with_leading_spaces.search(suffix):
            suffix = suffix[single_quote_with_leading_spaces.search(suffix).end() - 1:]

    prefix += TRIPLE_QUOTE
    suffix = "\n" + suffix
    return [prefix, suffix]


def build_docstring_infill_prompt(code: str,
        docstring_text: str = None,
        standardize_docstring: bool = True,
        ) -> List[str]:
    """Splits the function into a prompt prefix and suffix for the code -> docstring infilling task.

    Args:
        code: text of the function to split
        docstring_text: exact text of the docstring if it's already in the code string and should be stripped out

    Returns:
        list of len 2, splitting code into the part before and after the docstring
    """
    assert code.startswith("def") or code.startswith("async def"), "Must be a function definition"

    if docstring_text is not None:
        # note that we will infill using whatever docstring quote used originally in the function (could be """, ''', #, ', ")
        prompt_prefix = code[:code.index(docstring_text)]
        prompt_suffix = code[code.index(docstring_text) + len(docstring_text):]
    else:
        function_def = code[:code.index(":") + 1]
        body = code[code.index(":") + 1:]
        prompt_prefix = f"{function_def}\n{SPACES4}{TRIPLE_QUOTE} "
        prompt_suffix = " {TRIPLE_QUOTE}\n{body}"

    if standardize_docstring:
        prompt_prefix, prompt_suffix = standardize_docstring_prompt(prompt_prefix, prompt_suffix)

    prompt_suffix += f"\n{EOF}"
    return [prompt_prefix, prompt_suffix]

def build_systematic_infill_prompt(original_prompt: str, code: str, num_before: int, num_after: int) -> Tuple[List[str], str]:
    """Creates a prompt with given number of lines before and after to test infill systematically.
    
    Returns:
        prompt_parts (List[str]): list of len 2 [prefix, suffix]
        missing_lines (str): missing part to infill"""
    code_lines = code.split("\n")
    assert num_before + num_after < len(code_lines)
    assert original_prompt[-1] == "\n"
    prefix = "\n".join(code_lines[:num_before])
    suffix = "\n".join(code_lines[len(code_lines) - num_after:])
    missing_lines = "\n".join(code_lines[num_before:len(code_lines) - num_after])

    assert len(prefix.split("\n")) == num_before or (num_before == 0 and len(prefix) == 0)
    assert len(suffix.split("\n")) == num_after or (num_after == 0 and len(suffix) == 0)

    prompt_prefix = original_prompt + prefix
    if not prompt_prefix.endswith("\n"):
        prompt_prefix += "\n"

    return [prompt_prefix, suffix], missing_lines

def truncate_docstring_infill(infill: str) -> str:
    """Truncates an infill to the docstring text, removing extraneous generation output (e.g. additional functions).

    Note: assumes that there's no ' or " within the valid docstring
    """
    infill = infill.strip()
    # try to figure out where the end of the comment is
    if TRIPLE_QUOTE in infill:
        infill = infill[:infill.index(delim)]
    infill = infill.strip()
    return infill

def truncate_num_lines(infill: str, max_num_lines: int = 1) -> str:
    """Truncates infill to up to max number of lines."""
    infill_lines = stripped_line_split(infill)

    return "\n".join(infill_lines[:max_num_lines])

def stripped_line_split(text):
    return text.strip("\n").split("\n")

def truncate_overlap(infill, suffix, minimum_num_characters=None, minimum_num_suffix_lines=1):
    if minimum_num_characters is None:
        non_empty_suffix_lines = [l.strip() for l in suffix.strip("\n") if l.strip()]
        minimum_num_characters = sum(len(l) for l in non_empty_suffix_lines[:minimum_num_suffix_lines])
    for i in range(len(infill), minimum_num_characters, -1):
        if infill[-i:] == suffix[:i]:
            return infill[:-i]
    return infill

def read_file(filename):
    if filename.endswith(".json"):
        with open(filename) as f:
            return [json.loads(line) for line in f]
    elif filename.endswith(".pkl"):
        return unpickle(filename)
    else:
        raise NotImplementedError()

def all_equal(iterable):
    iterable = list(iterable)
    return all(iterable[0] == x for x in iterable)

def unpickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def dump_git_status(out_file=sys.stdout, exclude_file_patterns=['*.ipynb', '*.th', '*.sh', '*.txt', '*.json', '*.out']):
    subprocess.call('git rev-parse HEAD', shell=True, stdout=out_file)
    exclude_string = ' '.join("':(exclude){}'".format(f) for f in exclude_file_patterns)
    subprocess.call('git --no-pager diff -- . {}'.format(exclude_string), shell=True, stdout=out_file)

def dump_version_info(out_file=sys.stdout):
    try:
        print("fairseq version:", file=out_file)
        import fairseq
        print(fairseq.__version__, file=out_file)
        print(fairseq.__file__, file=out_file)
    except:
        print("fairseq not found", file=out_file)
