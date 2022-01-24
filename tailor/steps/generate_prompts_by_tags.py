from munch import Munch
from typing import Any, List, NamedTuple, Optional
from tango.step import Step

from tailor.steps.get_srl_tags import ProcessedSentence
from tailor.common.latest_utils import gen_prompts_by_tags, get_unique_tags


class PromptObject(NamedTuple):
    """
    TODO
    """

    prompt: Optional[str] = None
    answer: Optional[str] = None
    meta: Optional[Munch] = None  # TODO: use a PromptMeta abstraction.


def _munch_to_prompt_object(prompt_munch: Munch):
    return PromptObject(
        prompt=prompt_munch.prompt, answer=prompt_munch.answer, meta=prompt_munch.meta
    )


@Step.register("generate-prompts-by-tags")
class GeneratePromptsByTags(Step):
    """
    Generate intermediate prompts for all sentences.
    These are to be computed just once, and all actual prompts
    are constructed from these intermediate ones.
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        processed_sentences: List[ProcessedSentence],
        criteria_func: Optional[Any] = None,  # TODO
        **intermediate_prompt_kwargs,
    ) -> List[List[PromptObject]]:
        all_prompts = []
        for processed in processed_sentences:
            sentence_prompts = []
            for tags in processed.get_tags_list():  # TODO: apply criteria func?
                args_to_blank = get_unique_tags(tags)
                tags_prompt = gen_prompts_by_tags(
                    processed.spacy_doc,
                    frameset_id=None,
                    raw_tags=tags,
                    args_to_blank=args_to_blank,
                    return_prompt_type="concrete",  # TODO: when do we actually want all/sparse?
                    **intermediate_prompt_kwargs,
                )

                if isinstance(tags_prompt, List):
                    tags_prompt = [_munch_to_prompt_object(munch) for munch in tags_prompt]
                elif isinstance(tags_prompt, Munch):
                    tags_prompt = _munch_to_prompt_object(tags_prompt)
                else:
                    raise TypeError(
                        f"Unrecognized type {type(tags_prompt)} for generated intermediate prompt."
                    )
                sentence_prompts.append(tags_prompt)
            all_prompts.append(sentence_prompts)
        return all_prompts
