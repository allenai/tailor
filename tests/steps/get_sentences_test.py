import pytest
import datasets
from tailor.common.testing import TailorTestCase
from tailor.steps.get_sentences import GetSentences


class TestGetSpacyModel(TailorTestCase):
    def test_run(self):
        dataset = datasets.load_from_disk(str(self.FIXTURES_ROOT / "data" / "snli_snippet"))
        step = GetSentences()
        result = step.run(dataset=dataset, key="premise")

        assert len(result) == len(dataset) == 100
        assert isinstance(result, list)
        assert isinstance(result[0], str)

    def test_run_invalid_key(self):
        dataset = datasets.load_from_disk(str(self.FIXTURES_ROOT / "data" / "snli_snippet"))
        step = GetSentences()

        with pytest.raises(KeyError):
            step.run(dataset=dataset, key="invalid_key")

    def test_run_start_end_idx(self):
        dataset = datasets.load_from_disk(str(self.FIXTURES_ROOT / "data" / "snli_snippet"))
        step = GetSentences()

        result = step.run(dataset=dataset, key="premise", start_idx=10, end_idx=50)
        assert len(result) == 40
