from typing import Any, List, Optional, Tuple

import torch
from tqdm import tqdm
from munch import Munch
from tango.step import Step
from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines import Text2TextGenerationPipeline
from transformers.tokenization_utils import PreTrainedTokenizerBase

from tailor.common.abstractions import GeneratedPrompt, ProcessedSentence, PromptObject
from tailor.common.filters.perplex_filter import compute_delta_perplexity, load_perplex_scorer
from tailor.common.utils import SpacyModelType
from tailor.common.utils.generate_utils import compute_edit_ops
from tailor.common.utils.head_prompt_utils import BadGenerationError, parse_filled_prompt
from tailor.common.utils.model_utils import generate_and_clean_batch, load_generator


def _munch_to_generated_prompt(
    prompt_munch: Munch,
    name: Optional[str] = None,
    description: Optional[str] = None,
    perplexities: Optional[Munch] = None,
):
    return GeneratedPrompt(
        prompt_no_header=prompt_munch.prompt_no_header,
        sentence=prompt_munch.sentence,
        clean_sentence=prompt_munch.clean_sentence,
        meta=prompt_munch.meta,
        annotations=prompt_munch.annotations,
        words=prompt_munch.words,
        vidx=prompt_munch.vidx,
        name=name,
        description=description,
        perplexities=perplexities,
    )


def _unflatten(flattened_list: List[Any], original_lengths: List[int]):
    unflattened = []
    start = 0
    for length in original_lengths:
        end = start + length
        unflattened.append(flattened_list[start:end])
        start = end
    return unflattened


# @Step.register("generate-from-prompts")
class GenerateFromPrompts(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    VERSION = "05"

    def run(
        self,
        processed_sentences: List[ProcessedSentence],
        prompts: List[List[PromptObject]],
        spacy_model: SpacyModelType,
        generator: Optional[Text2TextGenerationPipeline] = None,
        generation_batch_size: int = 256,
        num_perturbations: int = 1,
        compute_perplexity: bool = False,
        perplex_scorer: Optional[Tuple[PreTrainedModel, PreTrainedTokenizerBase]] = None,
        **generation_kwargs,
    ) -> List[List[GeneratedPrompt]]:

        generator = generator or load_generator()

        is_cuda = torch.cuda.is_available()
        if compute_perplexity:
            if perplex_scorer:
                perplex_scorer = Munch(
                    model=perplex_scorer[0], tokenizer=perplex_scorer[1]
                )  # backwards compatibility.
            else:
                perplex_scorer = load_perplex_scorer(is_cuda=is_cuda)

        all_sentences = []

        assert len(prompts) == len(processed_sentences)

        batch_prompts = []
        sentence_lengths = []
        perturb_names = []
        perturb_descriptions = []

        all_prompts_flattened = []
        for idx, sentence in enumerate(processed_sentences): # TODO: add tqdm 
            batch_prompts += prompts[idx]
            sentence_lengths.append(len(prompts[idx]))  # number of prompts per sentence.
            perturb_names.append(
                [pr.name for pr in prompts[idx]]
            )  # number of prompts per sentence.
            perturb_descriptions.append([pr.description for pr in prompts[idx]])

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
        assert len(perturb_names) == len(processed_sentences)

        # TODO: separate this into a different step?
        for idx, generated_prompts in enumerate(all_generated_prompts):
            prompt_dicts = []
            assert (
                len(generated_prompts) == sentence_lengths[idx] == len(perturb_names[idx])
            )  # Sanity check
            for pdx, perturb in enumerate(generated_prompts):
                perturb_name = perturb_names[idx][pdx]
                perturb_desc = perturb_descriptions[idx][pdx]
                for raw_generated in perturb:
                    if raw_generated:
                        try:
                            prompt_dict = parse_filled_prompt(
                                raw_generated, nlp=spacy_model, is_compute_vidx=True
                            )
                        except BadGenerationError:
                            import traceback

                            #traceback.print_exc()
                            # TODO: add info that there was a bad generation.
                            continue

                        generated = prompt_dict.sentence
                        orig_doc = processed_sentences[idx].spacy_doc
                        if generated.lower() == orig_doc.text.lower():
                            continue
                        if compute_perplexity:
                            generated_doc = spacy_model(generated)
                            eop = compute_edit_ops(orig_doc, generated_doc)
                            if all([op.op == "equal" for op in eop]):
                                continue
                            perplexities = compute_delta_perplexity(
                                eop, perplex_scorer, is_cuda=is_cuda
                            )
                        else:
                            perplexities = None

                        prompt_dicts.append(
                            _munch_to_generated_prompt(
                                prompt_dict,
                                name=perturb_name,
                                description=perturb_desc,
                                perplexities=perplexities,
                            )
                        )
            all_sentences.append(prompt_dicts)

        return all_sentences
