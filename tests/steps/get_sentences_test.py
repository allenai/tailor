import pytest
import datasets
from tailor.common.testing import TailorTestCase
from tailor.steps.get_sentences import GetSentences


class TestGetSentences(TailorTestCase):
    def setup_method(self):
        super().setup_method()
        self.dataset = datasets.load_from_disk(str(self.FIXTURES_ROOT / "data" / "snli_snippet"))

    def test_run(self):

        step = GetSentences()
        result = step.run(dataset=self.dataset, key="premise")

        assert len(result) == len(dataset) == 100
        assert isinstance(result, list)
        assert isinstance(result[0], str)

    def test_run_with_dict(self):
        step = GetSentences()
        result = step.run(dataset=self.dataset.to_dict(), key="premise")

        assert len(result) == len(dataset) == 100
        assert isinstance(result, list)
        assert isinstance(result[0], str)

    def test_run_invalid_key(self):
        step = GetSentences()

        with pytest.raises(KeyError):
            step.run(dataset=self.dataset, key="invalid_key")

    def test_run_start_end_idx(self):
        step = GetSentences()

        result = step.run(dataset=self.dataset, key="premise", start_idx=10, end_idx=50)
        assert len(result) == 40
