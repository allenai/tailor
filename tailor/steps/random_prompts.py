import json
import os
from typing import Dict, List, Optional, Union

from tango.step import Step

from tailor.common.abstractions import ProcessedSentence, PromptObject
from tailor.common.utils import SpacyModelType
from tailor.common.utils.detect_perturbations import (
    DEFAULT_COMMON_KEYWORDS_PATH,
    detect_perturbations,
    get_common_keywords_by_tag,
)

PathOrStr = Union[os.PathLike, str]


def _load_default_common_keywords(
    common_keywords_json: PathOrStr = DEFAULT_COMMON_KEYWORDS_PATH,
) -> Dict[str, Dict]:
    with open(common_keywords_json, "r") as file_ref:
        common_keywords = json.load(file_ref)
    return common_keywords


@Step.register("get-common-keywords-by-tag")
class GetCommonKeywordsByTag(Step):
    """
    TODO (Alexis): Give a sense of what the common keywords mean here.
    """
    DETERMINISTIC = True
    CACHEABLE = False  # It's pretty fast

    def run(
        self,
        common_keywords_json: PathOrStr = DEFAULT_COMMON_KEYWORDS_PATH,
        data_path: Optional[PathOrStr] = None,
        spacy_model: Optional[SpacyModelType] = None,
        **kwargs,
    ) -> Dict[str, Dict]:
        if data_path is not None:
            return get_common_keywords_by_tag(nlp=spacy_model, **kwargs)

        common_keywords = _load_default_common_keywords(common_keywords_json)

        return common_keywords


@Step.register("generate-random-prompts")
class GenerateRandomPrompts(Step):
    """
    This step generates random prompts for perturbing the processed (spacy tokens and srl tags)
    sentences.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "generate-random-prompts".
    """

    DETERMINISTIC = False
    CACHEABLE = True

    def run(
        self,
        processed_sentences: List[ProcessedSentence],
        common_keywords_by_tag: Optional[Dict[str, Dict]] = None,
    ):
        """
        Parameters
        ----------

        processed_sentences : :class:`List[ProcessedSentence]`
            The list of processed sentences. See output of :class:`GetSRLTags`.
        common_keywords_by_tag : :class:`Dict[str, Dict]`
            See output of :class:`GetCommonKeywordsByTag`.
        """
        common_keywords_by_tag = common_keywords_by_tag or _load_default_common_keywords()
        all_prompts = []
        for sentence in processed_sentences:
            candidates = detect_perturbations(
                sentence.spacy_doc,
                start=None,
                end=None,
                predicted=sentence.verbs,
                common_keywords_by_tag=common_keywords_by_tag,
            )

            candidates = [
                PromptObject(prompt=prompt_munch.prompt, name=prompt_munch.constraint_type)
                for prompt_munch in candidates
            ]

            all_prompts.append(candidates)
        return all_prompts
