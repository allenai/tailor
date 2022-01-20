from munch import Munch
from typing import Any, Dict, Iterable, List, Optional
from tango.step import Step

from tailor.steps.generate_prompts_by_tags import IntermediatePrompts
from tailor.steps.get_srl_tags import ProcessedSentence, get_unique_tags
from tailor.common.old_utils import gen_prompts_by_tags
from tailor.common.perturb_function import PerturbFunction

from tailor.common.old_utils import *
from tailor.common.perturbation_controls import *


def is_equal_headers(p1, p2):
    """Helper function check for equality of headers between two prompts
    Useful for making sure that edited prompts are different
    Args:
        p1 (str): prompt
        p2 (str): prompt
    Returns:
        bool: Whether the two prompts have equal headers
    """
    return extract_header_from_prompt(p1)[0] == extract_header_from_prompt(p2)[0]


def get_unique_prompts(prompts):
    """Helper function to get unique prompts given list of prompts
    Helpful when we care about a looser notion of equality than exact string equality
    Calls is_equal_prompts() to check equality of prompts
    """
    prompt_set = []
    for p in prompts:
        if not any(is_equal_prompts(p, exist_p) for exist_p in prompt_set):
            prompt_set.append(p)
    return prompt_set


def is_equal_prompts(p1, p2):
    """Helper function check for equality of two prompts
    Insensitive to differences in space and punctuation in context
    Useful for making sure that edited prompts are different
    Args:
        p1 (str): prompt
        p2 (str): prompt
    Returns:
        bool: Whether the two prompts have equal prompts
    """

    def remove_punctuation(s):
        return re.sub(r"[.!?,-]", "", s)

    p1_head, p1_context = extract_header_from_prompt(p1)
    p2_head, p2_context = extract_header_from_prompt(p2)
    p1_context = remove_punctuation(p1_context).replace(" ", "").strip()
    p2_context = remove_punctuation(p2_context).replace(" ", "").strip()
    return p1_head.strip() == p2_head.strip() and p1_context.strip() == p2_context.strip()


"""
Thoughts: General case, you deal with one field: ie. just premise in nli.
Sometimes you want something like: do something to question based on context in qa.
So, another type of step for such cases where you take 2 processed sentences, and
determine which one is A and B, respectively.
"""


@Step.register("perturb-prompt-with-intermediate")
class PerturbPromptWithIntermediate(Step):
    """
    TODO
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        intermediate_prompts: IntermediatePrompts,
        processed_sentences: Iterable[ProcessedSentence],
        perturb_fn: PerturbFunction,
        **perturb_fn_kwargs,
    ):
        intermediate_prompts = intermediate_prompts.intermediate
        perturbations = []
        assert len(intermediate_prompts) == len(processed_sentences)
        for idx, intermediate_prompt in enumerate(intermediate_prompts):
            processed_sentence = processed_sentences[idx]

            # print(intermediate_prompt)
            # import ipdb
            # ipdb.set_trace()
            perturbations.append(
                perturb_fn(processed_sentence, intermediate_prompt, **perturb_fn_kwargs)
            )

        return perturbations


@Step.register("perturb-prompt")
class PerturbPrompt(Step):
    """
    TODO
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        processed_sentences: Iterable[ProcessedSentence],
        intermediate_prompt_kwargs: Dict[str, Any],
        perturb_fn: PerturbFunction,
        criteria_func: Optional[Any] = None,
        **perturb_fn_kwargs,
    ):
        all_prompts = []
        for processed in processed_sentences:
            sentence_prompts = []
            # TODO: allow user-defined option for extract_relative_clauses (criteria).
            # Each verb has a list of tags.
            for tags in processed.get_tags_list():  # apply the criteria for perturbation
                args_to_blank = get_unique_tags(tags)
                tags_prompt = gen_prompts_by_tags(
                    processed.spacy_doc,
                    frameset_id=None,
                    raw_tags=tags,
                    args_to_blank=args_to_blank,
                    **intermediate_prompt_kwargs,
                    # return_prompt_type=return_prompt_type,
                    # nblanks=nblanks,
                    # keyword_str=keyword_str,
                )

                prompt = perturb_fn(processed.spacy_doc, tags_prompt, tags)
                if prompt is not None:
                    sentence_prompts.append(prompt)
            sentence_prompts = get_unique_prompts(sentence_prompts)
            all_prompts.append(sentence_prompts)
        return all_prompts
