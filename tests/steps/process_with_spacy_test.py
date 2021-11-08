import os
import subprocess

import spacy

from tailor.common.testing import TailorTestCase
from tailor.steps.process_with_spacy import ProcessWithSpacy


class TestProcessWithSpacy(TailorTestCase):
    def test_perturbation(self):
        # spacy_model = spacy.load("en_core_web_sm")
        step = ProcessWithSpacy()
        result = step.run(
            inputs=["Hi this is a test", "Sample input text"], spacy_model_name="en_core_web_sm"
        )
        assert isinstance(result[0], spacy.tokens.doc.Doc)
        assert isinstance(result[1], spacy.tokens.doc.Doc)

    def test_deterministic_experiment(self):
        cmd = [
            "tango",
            "run",
            str(self.FIXTURES_ROOT / "experiment.jsonnet"),
            "-i",
            "tailor.steps.process_with_spacy",
            "-d",
            str(self.TEST_DIR),
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0
        assert len(os.listdir(self.TEST_DIR / "step_cache")) == 1

        # Running again shouldn't create any more directories.
        result = subprocess.run(cmd)
        assert result.returncode == 0
        assert len(os.listdir(self.TEST_DIR / "step_cache")) == 1
