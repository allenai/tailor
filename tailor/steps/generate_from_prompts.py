from munch import Munch
from typing import Iterable, List, Optional, NamedTuple
from transformers.pipelines import Text2TextGenerationPipeline

from tango.step import Step

from tailor.steps.get_srl_tags import ProcessedSentence

from tailor.common.latest_utils import parse_filled_prompt
from tailor.common.util import SpacyModelType

from tailor.steps.perturb_prompt import PromptObject

from tailor.common.model_utils import generate_and_clean_batch, load_generator

# Temporary wrapper to deal with Munch/Params issue.
class GeneratedPromptDict(NamedTuple):

    prompt_dict: Munch


@Step.register("generate-from-prompts")
class GenerateFromPrompts(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        processed_sentences: Iterable[ProcessedSentence],
        prompts: List[List[PromptObject]],
        spacy_model: SpacyModelType,
        generator: Optional[Text2TextGenerationPipeline] = None,
        num_perturbations: int = 3,
        **generation_kwargs,
    ) -> List[List[GeneratedPromptDict]]:

        generator = generator or load_generator()

        # TODO: make more efficient by flattening/unflattening and using batches for generation.
        all_sentences = []

        assert len(prompts) == len(processed_sentences)

        for idx, sentence in enumerate(processed_sentences):
            prompt_list = [p.prompt for p in prompts[idx]]  # list of str prompts
            generated_prompts = generate_and_clean_batch(
                prompts=prompt_list,
                generator=generator,
                n=num_perturbations,
                is_clean_verb_prefix=False,
                **generation_kwargs,
            )

            prompt_dicts = []
            orig_doc = sentence.spacy_doc

            if generated_prompts:
                merged = [val for sublist in generated_prompts for val in sublist]
                # validate
                for raw_generated in merged:
                    try:
                        prompt_dict = parse_filled_prompt(
                            raw_generated, nlp=spacy_model, is_compute_vidx=True
                        )
                    except:
                        continue
                    prompt_dicts.append(GeneratedPromptDict(prompt_dict))

            all_sentences.append(prompt_dicts)

        return all_sentences
