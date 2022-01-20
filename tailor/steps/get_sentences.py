from typing import Dict, Iterable, List, Optional
from tango.step import Step
import datasets


@Step.register("get-sentences")
class GetSentences(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        dataset: datasets.Dataset,
        key: str,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Iterable[str]:

        return dataset[key][start_idx:end_idx]  # None works as expected.
