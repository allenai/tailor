from typing import Any, Dict, List, Optional

import datasets
from tango.step import Step


@Step.register("convert-dataset-to-dict")
class ConvertDatasetToDict(Step):
    """
    This step converts a Huggingface dataset to dict, and optionally returns an subset.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "convert-dataset-to-dict".
    """

    DETERMINISTIC = True
    CACHEABLE = False

    def run(
        self,
        dataset: datasets.Dataset,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Dict[str, List[Any]]:
        """
        Parameters
        ----------

        dataset : :class:`datasets.Dataset`
            The huggingface dataset to convert.
        start_idx : :class:`int`, optional
            The start index for returning a subset of the data.
            Default is None.
        end_idx : :class:`int`, optional
            The end index for returning a subset of the data.
            Default is None.

        Returns
        -------

        :class:`Dict[str, List[Any]]`
            The dataset in dict format.
        """

        # keys = dataset.features
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
    """
    This step converts a dict to a Huggingface dataset.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "convert-dict-to-dataset".
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, data_dict: Dict[str, List[Any]]):
        return datasets.Dataset.from_dict(data_dict)
