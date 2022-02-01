from munch import Munch
from typing import Iterable, List, Optional
from transformers.pipelines import Text2TextGenerationPipeline

from tango.step import Step

from tailor.common.latest_utils import parse_filled_prompt
from tailor.common.util import SpacyModelType

from tailor.common.abstractions import ProcessedSentence, PromptObject, GeneratedPrompt

from tailor.common.model_utils import generate_and_clean_batch, load_generator


def _munch_to_generated_prompt(prompt_munch: Munch, name: Optional[str] = None):
    return GeneratedPrompt(
        prompt_no_header=prompt_munch.prompt_no_header,
        sentence=prompt_munch.sentence,
        meta=prompt_munch.meta,
        annotations=prompt_munch.annotations,
        words=prompt_munch.words,
        vidx=prompt_munch.vidx,
        name=name,
    )


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
    ) -> List[List[GeneratedPrompt]]:

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
            # orig_doc = sentence.spacy_doc

            if generated_prompts:
                merged = [val for sublist in generated_prompts for val in sublist]
                # validate
                for raw_generated in merged:
                    try:
                        prompt_dict = parse_filled_prompt(
                            raw_generated, nlp=spacy_model, is_compute_vidx=True
                        )
                    except:
                        import traceback
                        traceback.print_exc()
                        continue
                    prompt_dicts.append(_munch_to_generated_prompt(prompt_dict, raw_generated))

            all_sentences.append(prompt_dicts)

        return all_sentences
