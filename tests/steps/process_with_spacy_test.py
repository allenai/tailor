import pytest

from tailor.common.testing import TailorTestCase
from tailor.steps.process_with_spacy import (
    GetSpacyModel,
    ProcessWithSpacy,
    SpacyDoc,
    SpacyModelType,
    get_spacy_model,
)


class TestGetSpacyModel(TailorTestCase):
    def test_run(self):
        step = GetSpacyModel()
        result = step.run()

        assert isinstance(result, SpacyModelType)

        out = result("Hi this is a test.")
        assert isinstance(out, SpacyDoc)

        assert len(out) == 6

    def test_run_with_white_space_tokenizer(self):
        step = GetSpacyModel()
        result = step.run(use_white_space_tokenizer=True)

        assert isinstance(result, SpacyModelType)

        out = result("Hi this is a test.")
        assert isinstance(out, SpacyDoc)

        assert len(out) == 5

    def test_run_with_spacy_kwargs(self):
        step = GetSpacyModel()
        result = step.run(parse=True)

        assert isinstance(result, SpacyModelType)

        out = result("Hi this is a test.")
        assert isinstance(out, SpacyDoc)

        assert (len([chunk for chunk in out.noun_chunks])) == 1

        step = GetSpacyModel()
        result = step.run(parse=False)

        assert isinstance(result, SpacyModelType)

        out = result("Hi this is a test.")
        assert isinstance(out, SpacyDoc)

        with pytest.raises(ValueError):
            [chunk for chunk in out.noun_chunks]


class TestProcessWithSpacy(TailorTestCase):
    def test_perturbation(self):
        step = ProcessWithSpacy()
        spacy_model = get_spacy_model("en_core_web_sm")
        result = step.run(
            sentences=["Hi this is a test.", "Sample input text"], spacy_model=spacy_model
        )
        assert isinstance(result[0], SpacyDoc)
        assert isinstance(result[1], SpacyDoc)

        assert len(result[0]) == 6
