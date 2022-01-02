import os
import subprocess

import spacy

from tailor.common.testing import TailorTestCase
from tailor.steps.get_srl_tags import GetSRLPredictions, GetSRLTags


class TestGetSRLTags(TailorTestCase):
    def test_perturbation(self):
        step = GetSRLPredictions()
        result = step.run(sentences=["Hi this is a test", "Sample input text"])

        assert len(result) == 2
        assert "verbs" in result[0]
        assert "tags" in result[0]["verbs"][0]

        step = GetSRLTags()
        tags = step.run(result)

        assert len(tags) == 2
