import os
import subprocess

import spacy

from tailor.common.testing import TailorTestCase
from tailor.steps.process_with_spacy import ProcessWithSpacy
from tailor.steps.get_srl_tags import GetSRLPredictions, SRLTags, ProcessedSentence


class TestGetSRLTags(TailorTestCase):
    # def test_perturbation(self):
    #     step = GetSRLPredictions()
    #     srl_predictions = step.run(sentences=["Hi this is a test", "Sample input text", "this is a test, and it is passing well"])

    #     # assert len(srl_predictions) == 2
    #     # assert "verbs" in srl_predictions[0]
    #     # assert "tags" in srl_predictions[0]["verbs"][0]

    #     print(srl_predictions)
    #     assert False

    #     step = GetSRLTags()
    #     tags = step.run(srl_predictions=srl_predictions)

    #     assert len(tags) == 2
    #     # assert tags == [[['ARG2', 'V', 'ARGM-DIS', 'ARG1']], []]

    def test_step(self):

        step = ProcessWithSpacy()
        spacy_outs = step.run(sentences=["Hi this is a test.", "Sample input text"])

        step = SRLTags()
        tagged = step.run(spacy_outputs=spacy_outs)

        # assert tagged == [[['B-ARGM-DIS', 'B-ARG1', 'B-V', 'B-ARG2', 'I-ARG2', 'O']], []]
        assert len(tagged) == 2

        assert isinstance(tagged[0], ProcessedSentence)
        assert tagged[0].get_tags_list() == [
            ["B-ARGM-DIS", "B-ARG1", "B-V", "B-ARG2", "I-ARG2", "O"]
        ]
