from typing import Iterable

from tango.step import Step

from tailor.common.util import get_spacy_model


@Step.register("process-with-spacy")
class ProcessWithSpacy(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    VERSION = "001"

    def run(self, inputs: Iterable[str], spacy_model_name: str) -> Iterable[str]:
        # The model is cached, so no worries about reloading.
        spacy_model = get_spacy_model(spacy_model_name)
        output = []
        for string in inputs:
            output.append(spacy_model(string))
        return output
