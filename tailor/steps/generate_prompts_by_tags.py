from munch import Munch
from typing import Dict, Iterable, List, NamedTuple, Optional
from tango.step import Step

from tailor.steps.get_srl_tags import ProcessedSentence, get_unique_tags
from tailor.common.old_utils import gen_prompts_by_tags

# TODO: temporary fix to sidestep tango config being overzealous with Params.
class IntermediatePrompts(NamedTuple):
    intermediate: List[List[Munch]]


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
        processed_sentences: Iterable[ProcessedSentence],
        keyword_str: str = "NOUN_CHUNKS,RANDOM_SUBTREES,EXACT,PREFIX",
        short_args_to_blank: Optional[List[str]] = None,  # See: tag_utils. CORE/Non-CORE tags.
        nblanks: Optional[int] = None,
        return_sequence: bool = True,  # TODO: is this required?
        is_blank_aux: bool = True,
        no_empty_blanks_at_start: bool = False,
        p_overblank: float = 0.0,
        return_prompt_type: str = "concrete",  # SPECIFICITY; concrete seems to be most common.
    ) -> IntermediatePrompts:
        """
        Arguments:
            TODO
            We save ALL intermediate prompts for all tags.
        """
        all_prompts = []
        for processed in processed_sentences:
            sentence_prompts = []
            for (
                tags
            ) in (
                processed.get_tags_list()
            ):  # TODO: allow user-defined option for extract_relative_clauses.
                # Each verb has a list of tags.
                args_to_blank = get_unique_tags(tags)
                tags_prompt = gen_prompts_by_tags(
                    processed.spacy_doc,
                    None,
                    tags,
                    return_prompt_type=return_prompt_type,
                    nblanks=nblanks,
                    args_to_blank=args_to_blank,
                    keyword_str=keyword_str,
                )
                sentence_prompts.append(tags_prompt)
            all_prompts.append(sentence_prompts)
        return IntermediatePrompts(all_prompts)
