from typing import Dict, List, NamedTuple, Optional

from allennlp.predictors import Predictor
from tango.step import Step

from tailor.common.util import get_srl_tagger
from tailor.steps.process_with_spacy import SpacyDoc


class ProcessedSentence(NamedTuple):

    sentence: str
    spacy_doc: SpacyDoc
    verbs: List[Dict]  # Dict: {"verb": str, "tags": List[str]}

    def get_tags_list(self):
        return [verb_dict["tags"] for verb_dict in self.verbs]


@Step.register("get-srl-tags")
class GetSRLTags(Step):
    """
    TODO: Docs
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        spacy_outputs: List[SpacyDoc],
        srl_tagger: Optional[Predictor] = None,
    ) -> List[ProcessedSentence]:  # Multiple verbs for each sentence.
        srl_tagger = srl_tagger or get_srl_tagger()
        outputs = []
        for spacy_doc in spacy_outputs:
            sentence = " ".join([token.text for token in spacy_doc])
            srl_prediction = srl_tagger.predict_json(
                {"sentence": sentence}
            )  # TODO: use batches for efficiency.

            outputs.append(ProcessedSentence(sentence, spacy_doc, srl_prediction["verbs"]))

        return outputs
