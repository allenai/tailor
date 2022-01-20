from munch import Munch
from typing import Dict, Iterable, List, NamedTuple, Optional
from tango.step import Step

from tailor.steps.get_srl_tags import ProcessedSentence

from tailor.common.latest_utils import parse_filled_prompt
from tailor.common.util import SpacyModelType

# from tailor.common.model_utils import generate_and_clean_batch, load_generator
from tailor.common.generate_utils import compute_edit_ops, generate_and_clean_batch, load_generator


# class GeneratedPrompt(NamedTuple):

#     generated: str
#     clean_generated: Optional[str] = None


@Step.register("generate-from-prompts")
class GenerateFromPrompts(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        processed_sentences: Iterable[ProcessedSentence],
        prompts: List[List[str]],
        spacy_model: SpacyModelType,
        num_perturbations: int = 3,
        perplex_thred: Optional[int] = None,
        **generation_kwargs,
    ):

        generator = load_generator()  # make it a step output?

        # TODO: make more efficient by flattening/unflattening and using batches for generation.
        all_sentences = []

        assert len(prompts) == len(processed_sentences)

        for idx, sentence in enumerate(processed_sentences):
            prompt_list = prompts[idx]  # list of str prompts
            generated_prompts = generate_and_clean_batch(
                prompts=prompt_list,
                generator=generator,
                n=num_perturbations,
                is_clean_verb_prefix=False,
                **generation_kwargs,
            )

            validated_set = []
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
                    generated = prompt_dict.sentence
                    if generated in validated_set or generated.lower() == orig_doc.text.lower():
                        continue
                    is_valid = True
                    # # TODO:
                    # generated_doc = spacy_model(generated)
                    # if perplex_thred is not None:
                    #     eop = compute_edit_ops(orig_doc, generated_doc)
                    #     pp = self._compute_delta_perplexity(eop)
                    #     is_valid = pp.pr_sent < perplex_thred and pp.pr_phrase < perplex_thred
                    # if is_valid:
                    #     predicted = self.srl_predict(generated_s)
                    #     prompt_dict = add_predictions_to_prompt_dict(prompt_dict, predicted)
                    #     is_valid = is_vaild and is_followed_ctrl(prompt_dict, generated_doc, self.spacy_processor)
                    if is_valid:
                        validated_set.append(generated)

            all_sentences.append(validated_set)
            # all_sentences.append(generated)

        return all_sentences
