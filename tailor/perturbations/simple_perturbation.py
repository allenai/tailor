from typing import Sequence

# from tango.common import DatasetDict
from tango.step import Step


@Step.register("simple-perturbation")
class SimplePerturbation(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    VERSION = "001"

    def run(self, inputs: Sequence[str]) -> Sequence[str]:
        output = []
        for string in inputs:
            output.append(string.lower() + "!")
        return output
