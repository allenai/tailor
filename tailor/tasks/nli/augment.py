import random
from tqdm import tqdm
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

        new_data: Dict[str, List[str]] = {
                "premise": [], 
                "hypothesis": [], 
                "label": [], 
                "perturbation_strategy": [],
                }
        # TODO: add sanity checks.
        assert len(dataset[perturbed_field]) == len(generated_prompt_dicts)
        
        for idx, generations in tqdm(enumerate(generated_prompt_dicts)):
            # Look at unique generations if some perturbations result in equivalent generations
            # (eg SwapCoreWithContext and SwapCoreWithoutContext might)
            seen_generations = set() 
            unique_generations = [seen_generations.add(gen.clean_sentence) or gen for gen in generations \
                    if gen.clean_sentence not in seen_generations]

            num_augments = min(max_augment_per_instance, len(unique_generations))
            for generation in random.sample(unique_generations, num_augments):
                if generation.description == "preserves_meaning":
                    label = "entailment"
                else:
                    label = "neutral"

                if perturbed_field == "premise":
                    premise = dataset["premise"][idx]
                else:
                    premise = dataset["hypothesis"][idx]

                hypothesis = generation.clean_sentence
                new_data["premise"].append(premise)
                new_data["hypothesis"].append(hypothesis)
                new_data["label"].append(label)
                new_data["perturbation_strategy"].append(generation.name)

        return new_data
