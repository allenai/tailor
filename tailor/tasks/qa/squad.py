import glob
import json
import os
from typing import Dict, List, Optional

from tango.common import DatasetDict
from tango.step import Step

from tailor.common.util import get_spacy_model


@Step.register("load-squad")
class LoadSquad(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, data_dir: str, splits: Optional[List[str]] = None) -> DatasetDict:

        if not splits:
            splits = []
            for filename in glob.glob(os.path.join(data_dir, "*.json")):
                splits.append(filename.replace(".json", ""))

        split_data: Dict[str, List] = {}
        for split in splits:
            with open(os.path.join(data_dir, split + ".json")) as file_ref:
                data = json.load(file_ref)

            split_data[split] = data

        return DatasetDict(split_data)


@Step.register("process-squad-with-spacy")  # TODO: change name?
class ProcessSquadWithSpacy(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        dataset_dict: DatasetDict,
        spacy_model_name: str,
        keys_to_process: Optional[List[str]] = None,
    ) -> DatasetDict:
        # The model is cached, so no worries about reloading.
        spacy_model = get_spacy_model(spacy_model_name)
        split_data: Dict[str, List] = {}

        for split in dataset_dict:
            dataset = dataset_dict[split]
            keys_to_process = keys_to_process or dataset[0].keys()

            split_data[split] = []

            for instance in dataset:
                processed_instance = {}
                for key in dataset[0].keys():
                    if key in keys_to_process:
                        processed_instance[key] = spacy_model(instance[key])
                    else:
                        processed_instance[key] = instance[key]
                split_data[split].append(processed_instance)

        return DatasetDict(split_data)
