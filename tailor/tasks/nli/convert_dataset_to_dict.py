from typing import Any, Dict, List, Optional

import datasets
from tango.step import Step


@Step.register("convert-dataset-to-dict")
class ConvertDatasetToDict(Step):
    """
    TODO
    """

    DETERMINISTIC = True
    CACHEABLE = False

    def run(
        self,
        dataset: datasets.Dataset,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Dict[str, List[Any]]:

        data_dict = dataset[start_idx:end_idx]
        return data_dict

        # instances = []

        # for idx in range(len(data_dict[keys[0]])):
        #     instance = {}
        #     for key in keys:
        #         instance[key] = data_dict[key][idx]
        #     instances.append(instance)

        # return instances


@Step.register("convert-dict-to-dataset")
class ConvertDictToDataset(Step):

    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, data_dict: Dict[str, List[Any]]):
        return datasets.Dataset.from_dict(data_dict)
