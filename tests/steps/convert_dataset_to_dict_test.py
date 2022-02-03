import datasets

from tailor.common.testing import TailorTestCase
from tailor.steps.convert_dataset_to_dict import ConvertDatasetToDict, ConvertDictToDataset


class TestConvertDatasetDict(TailorTestCase):
    def setup_method(self):
        super().setup_method()
        self.dataset = datasets.load_from_disk(str(self.FIXTURES_ROOT / "data" / "snli_snippet"))
        self.data_dict = self.dataset.to_dict()

    def test_convert_dataset_to_dict(self):
        step = ConvertDatasetToDict()
        result = step.run(dataset=self.dataset)

        assert isinstance(result, dict)

    def test_convert_dict_to_dataset(self):
        step = ConvertDictToDataset()
        result = step.run(data_dict=self.data_dict)

        assert isinstance(result, datasets.Dataset)

    def test_start_end_idx(self):
        step = ConvertDatasetToDict()

        result = step.run(dataset=self.dataset, start_idx=10, end_idx=50)
        assert len(result["premise"]) == 40
