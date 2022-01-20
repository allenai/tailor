from typing import Any, Dict, Iterable, List, Optional
from tango.step import Step

from tailor.steps.generate_prompts_by_tags import PromptObject
from tailor.steps.get_srl_tags import ProcessedSentence
from tailor.common.perturb_function import PerturbFunction, PerturbStringFunction

from tailor.common.latest_utils import (
    gen_prompts_by_tags,
    get_unique_tags,
    get_unique_prompts,
    gen_prompt_by_perturb_str,
    is_equal_headers,
)

"""
Thoughts: General case, you deal with one field: ie. just premise in nli.
Sometimes you want something like: do something to question based on context in qa.
So, another type of step for such cases where you take 2 processed sentences, and
determine which one is A and B, respectively.
"""


@Step.register("perturb-prompt-with-intermediate")
class PerturbPromptWithIntermediate(Step):
    """
    TODO: Docs
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        intermediate_prompts: List[List[PromptObject]],
        processed_sentences: Iterable[ProcessedSentence],
        perturb_fn: PerturbFunction,
        **perturb_fn_kwargs,
    ) -> List[List[str]]:
        perturbations = []
        assert len(intermediate_prompts) == len(processed_sentences)
        for idx, intermediate_prompt in enumerate(intermediate_prompts):
            processed_sentence = processed_sentences[idx]
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
        criteria_func: Optional[Any] = None,  # TODO.
        **perturb_fn_kwargs,
    ) -> List[List[str]]:
        all_prompts = []
        for processed in processed_sentences:
            sentence_prompts = []
            for tags in processed.get_tags_list():  # apply the criteria for perturbation
                args_to_blank = get_unique_tags(tags)
                tags_prompt = gen_prompts_by_tags(
                    processed.spacy_doc,
                    frameset_id=None,
                    raw_tags=tags,
                    args_to_blank=args_to_blank,
                    return_prompt_type="concrete",
                    **intermediate_prompt_kwargs,
                )

                prompt = perturb_fn(processed.spacy_doc, tags_prompt, tags)
                if prompt is not None:
                    sentence_prompts.append(prompt)
            sentence_prompts = get_unique_prompts(sentence_prompts)
            all_prompts.append(sentence_prompts)
        return all_prompts


@Step.register("perturb-prompt-by-str-with-intermediate")
class PerturbPromptWithIntermediate(Step):
    """
    TODO: Docs
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        intermediate_prompts: List[List[PromptObject]],
        processed_sentences: Iterable[ProcessedSentence],
        perturb_str: str,
    ) -> List[List[str]]:
        perturbations = []
        assert len(intermediate_prompts) == len(processed_sentences)
        for idx, intermediate_prompt in enumerate(intermediate_prompts):
            processed_sentence = processed_sentences[idx]
            srl_tag_list = processed_sentence.get_tags_list()

            sentence_perturbations = []
            for tag_idx, tags_prompt in enumerate(intermediate_prompt):
                perturbed = gen_prompt_by_perturb_str(
                    processed_sentence.spacy_doc,
                    srl_tag_list[tag_idx],
                    perturb_str,
                    tags_prompt.meta,
                )

                if is_equal_headers(perturbed.prompt, tags_prompt.prompt):
                    prompt = None
                else:
                    prompt = perturbed.prompt

                sentence_perturbations.append(prompt)

            perturbations.append(sentence_perturbations)

        return perturbations


@Step.register("perturb-prompt-by-str")
class PerturbPromptByString(Step):
    """
    TODO: Docs
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        processed_sentences: Iterable[ProcessedSentence],
        intermediate_prompt_kwargs: Dict[str, Any],
        perturb_str: str,
    ) -> List[List[str]]:
        all_prompts = []
        for processed in processed_sentences:
            sentence_prompts = []
            for tags in processed.get_tags_list():  # apply the criteria for perturbation
                args_to_blank = get_unique_tags(tags)
                tags_prompt = gen_prompts_by_tags(
                    processed.spacy_doc,
                    frameset_id=None,
                    raw_tags=tags,
                    args_to_blank=args_to_blank,
                    return_prompt_type="concrete",
                    **intermediate_prompt_kwargs,
                )

                perturbed = gen_prompt_by_perturb_str(
                    processed.spacy_doc, tags, perturb_str, tags_prompt.meta
                )

                if is_equal_headers(perturbed.prompt, tags_prompt.prompt):
                    prompt = None
                else:
                    prompt = perturbed.prompt

                if prompt is not None:
                    sentence_prompts.append(prompt)
            sentence_prompts = get_unique_prompts(sentence_prompts)
            all_prompts.append(sentence_prompts)
        return all_prompts


@Step.register("perturb-prompt-with-function")
class PerturbPromptByString(Step):
    """
    TODO: Docs
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        processed_sentences: Iterable[ProcessedSentence],
        intermediate_prompt_kwargs: Dict[str, Any],
        perturb_func: PerturbStringFunction,
    ) -> List[List[str]]:
        all_prompts = []
        for processed in processed_sentences:
            sentence_prompts = []
            for tags in processed.get_tags_list():  # apply the criteria for perturbation
                args_to_blank = get_unique_tags(tags)
                tags_prompt = gen_prompts_by_tags(
                    processed.spacy_doc,
                    frameset_id=None,
                    raw_tags=tags,
                    args_to_blank=args_to_blank,
                    return_prompt_type="concrete",
                    **intermediate_prompt_kwargs,
                )

                perturb_str = perturb_func(tags_prompt.meta)
                # TODO: return an object with metadata info: which perturb func created the prompt.

                perturbed = gen_prompt_by_perturb_str(
                    processed.spacy_doc, tags, perturb_str, tags_prompt.meta
                )

                if is_equal_headers(perturbed.prompt, tags_prompt.prompt):
                    prompt = None
                else:
                    prompt = perturbed.prompt

                if prompt is not None:
                    sentence_prompts.append(prompt)
            sentence_prompts = get_unique_prompts(sentence_prompts)
            all_prompts.append(sentence_prompts)
        return all_prompts
