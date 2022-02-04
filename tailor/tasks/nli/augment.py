import random
from typing import Dict, List

from tango.step import Step

from tailor.common.abstractions import GeneratedPrompt


@Step.register("augment-nli")
class AugmentNLI(Step):
    def run(
        self,
        dataset: Dict[str, List[str]],
        perturbed_field: str,
        generated_prompt_dicts: List[List[GeneratedPrompt]],
        max_augment_per_instance: int = 2,
    ) -> Dict[str, List[str]]:

        new_data: Dict[str, List[str]] = {"premise": [], "hypothesis": [], "label": []}
        # TODO: add sanity checks.
        assert len(dataset[perturbed_field]) == len(generated_prompt_dicts)

        for idx, generations in enumerate(generated_prompt_dicts):
            num_augments = min(max_augment_per_instance, len(generations))
            for generation in random.sample(generations, num_augments):
                if generation.description == "preserves_meaning":
                    label = dataset["label"][idx]
                    if perturbed_field == "premise":
                        premise = generation.sentence
                        hypothesis = dataset["hypothesis"][idx]
                    else:
                        premise = dataset["premise"][idx]
                        hypothesis = generation.sentence
                    new_data["premise"].append(premise)
                    new_data["hypothesis"].append(hypothesis)
                    new_data["label"].append(label)

        return new_data
