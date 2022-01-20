import os
import subprocess

import spacy
from tango.common import DatasetDict
from tailor.common.testing import TailorTestCase
from tailor.steps.process_with_spacy import (
    ProcessWithSpacy,
    ProcessDatasetWithSpacy,
    GetSpacyModel,
    SpacyModelType,
    SpacyDoc,
)


class TestProcessWithSpacy(TailorTestCase):
    def test_perturbation(self):
        step = ProcessWithSpacy()
        result = step.run(
            sentences=["Hi this is a test.", "Sample input text"], spacy_model_name="en_core_web_sm"
        )
        assert isinstance(result[0].spacy_doc, SpacyDoc)
        assert isinstance(result[1].spacy_doc, SpacyDoc)

        assert len(result[0].spacy_doc) == 6

    def test_with_white_space(self):
        step = ProcessWithSpacy()
        result = step.run(
            sentences=["Hi this is a test."],
            use_white_space_tokenizer=True,
        )

        assert isinstance(result[0].spacy_doc, SpacyDoc)
        assert len(result[0].spacy_doc) == 5

    def test_spacy_model(self):
        step = GetSpacyModel()
        result = step.run()

        assert isinstance(result, SpacyModelType)

    def test_process_dataset_with_spacy(self):
        dataset = DatasetDict(
            {"train": [{"sentence": "Hi this is a test."}, {"sentence": "Sample input text"}]}
        )

        step = GetSpacyModel()
        model = step.run()

        step = ProcessDatasetWithSpacy()
        new_dataset = step.run(spacy_model=model, dataset_dict=dataset, key_to_process="sentence")

        assert isinstance(new_dataset["train"][0]["sentence"], SpacyDoc)

        step = ProcessDatasetWithSpacy()
        new_dataset = step.run(
            spacy_model=model,
            dataset_dict=dataset,
            key_to_process="sentence",
            processed_key_name="processed_sentence",
        )

        assert isinstance(new_dataset["train"][0]["sentence"], str)
        assert isinstance(new_dataset["train"][0]["processed_sentence"], SpacyDoc)

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
