from munch import Munch
from typing import Any, List, Optional
from transformers.pipelines import Text2TextGenerationPipeline

from tango.step import Step

from tailor.common.latest_utils import parse_filled_prompt, BadGenerationError
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


def _unflatten(flattened_list: List[Any], original_lengths: List[int]):
    unflattened = []
    start = 0
    for length in original_lengths:
        end = start + length
        unflattened.append(flattened_list[start:end])
        start = end
    return unflattened


@Step.register("generate-from-prompts")
class GenerateFromPrompts(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        processed_sentences: List[ProcessedSentence],
        prompts: List[List[PromptObject]],
        spacy_model: SpacyModelType,
        generator: Optional[Text2TextGenerationPipeline] = None,
        generation_batch_size: int = 256,
        num_perturbations: int = 3,
        **generation_kwargs,
    ) -> List[List[GeneratedPrompt]]:

        generator = generator or load_generator()

        all_sentences = []

        assert len(prompts) == len(processed_sentences)

        batch_prompts = []
        sentence_lengths = []

        all_prompts_flattened = []
        for idx, sentence in enumerate(processed_sentences):  # TODO: add tqdm
            batch_prompts += prompts[idx]
            sentence_lengths.append(len(prompts[idx]))  # number of prompts per sentence.

            if len(batch_prompts) >= generation_batch_size or idx == len(processed_sentences) - 1:
                prompt_list = [p.prompt for p in batch_prompts]  # list of str prompts
                batch_generated_prompts = generate_and_clean_batch(
                    prompts=prompt_list,
                    generator=generator,
                    n=num_perturbations,
                    is_clean_verb_prefix=False,
                    **generation_kwargs,
                )

                if not batch_generated_prompts:
                    batch_generated_prompts = []

                all_prompts_flattened += batch_generated_prompts
                batch_prompts = []

        all_generated_prompts = _unflatten(all_prompts_flattened, sentence_lengths)
        assert len(all_generated_prompts) == len(processed_sentences)

        for idx, generated_prompts in enumerate(all_generated_prompts):
            prompt_dicts = []
            assert len(generated_prompts) == sentence_lengths[idx]  # Sanity check
            merged = [val for sublist in generated_prompts for val in sublist]
            for raw_generated in merged:
                if raw_generated:
                    try:
                        prompt_dict = parse_filled_prompt(
                            raw_generated, nlp=spacy_model, is_compute_vidx=True
                        )
                    except BadGenerationError:
                        import traceback

                        traceback.print_exc()
                        # TODO: add info that there was a bad generation.
                        continue
                    prompt_dicts.append(_munch_to_generated_prompt(prompt_dict))
            all_sentences.append(prompt_dicts)

        return all_sentences
