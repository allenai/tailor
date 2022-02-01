from typing import List, Optional

from allennlp.predictors import Predictor
from tango.step import Step

from tailor.common.util import get_srl_tagger, predict_batch_srl
from tailor.steps.process_with_spacy import SpacyDoc
from tailor.common.abstractions import ProcessedSentence


@Step.register("get-srl-tags")
class GetSRLTags(Step):
    """
    This step applies the SRL tagger to the provided list of spacy docs.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "get-srl-tags".
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        spacy_outputs: List[SpacyDoc],
        srl_tagger: Optional[Predictor] = None,
    ) -> List[ProcessedSentence]:
        """
        Returns the list of sentences with SRL tags.

        Parameters
        ----------
        spacy_outputs : :class:`List[SpacyDoc]`
            The list of spacy docs for the list of sentences.
        srl_tagger : :class:`Predictor`, optional
            An AllenNLP predictor for srl tagging. The default is the `srl-bert` predictor.

        Returns
        -------
        :class:`List[ProcessedSentence]`
            The list of processed sentences with spacy docs and SRL tags.

        """
        srl_tagger = srl_tagger or get_srl_tagger()
        outputs = []
        sentences = [" ".join([token.text for token in spacy_doc]) for spacy_doc in spacy_outputs]
        srl_predictions = predict_batch_srl(sentences, srl_tagger)

        assert len(srl_predictions) == len(sentences) == len(spacy_outputs)

        for idx, srl_prediction in enumerate(srl_predictions):
            outputs.append(
                ProcessedSentence(sentences[idx], spacy_outputs[idx], srl_prediction["verbs"])
            )

        return outputs
