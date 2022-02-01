from typing import Dict, List
from tango.step import Step
from tailor.common.abstractions import ProcessedSentence
from tailor.common.utils import SpacyModelType
from tailor.common.utils.detect_perturbations import get_common_keywords_by_tag, detect_perturbations

@Step.register("get-common-keywords-by-tag")
class GetCommonKeywordsByTag(Step):
    DETERMINISTIC = True
    CACHEABLE = False  # It's pretty fast

    def run(
        self,
        spacy_model: SpacyModelType,
        **kwargs,
    ) -> Dict[str, Dict]:
        return get_common_keywords_by_tag(nlp=spacy_model, **kwargs)


@Step.register("generate-random-prompts")
class GenerateRandomPrompts(Step):

    DETERMINISTIC = False
    CACHEABLE = True

    def run(
        self,
        processed_sentences: List[ProcessedSentence],
        common_keywords_by_tag: Dict[str, Dict],
    ):
        all_prompts = []
        for sentence in processed_sentences:
            candidates = detect_perturbations(sentence.spacy_doc, start=None, end=None, predicted=sentence.verbs, common_keywords_by_tag=common_keywords_by_tag)
            all_prompts.append(candidates)
        return all_prompts
