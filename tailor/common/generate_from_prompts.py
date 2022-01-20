from munch import Munch
from typing import Dict, Iterable, List, NamedTuple, Optional
from tango.step import Step

from tailor.steps.get_srl_tags import ProcessedSentence, get_unique_tags

# from tailor.common.old_utils import gen_prompts_by_tags, parse_filled_prompt
from tailor.common.latest_utils import parse_filled_prompt
from tailor.common.util import get_spacy_model, SpacyDoc, SpacyModelType
from tailor.common.model_utils import generate_and_clean_batch, load_generator
from tailor.common.generate_utils import compute_edit_ops


class GeneratedPrompt(NamedTuple):

    generated: str
    clean_generated: Optional[str] = None


@Step.register("generate-from-prompts")
class GenerateFromPrompts(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        prompts: Iterable[List],
        # spacy_model: SpacyModelType, # probably don't need this.
        processed_sentences: Optional[Iterable[ProcessedSentence]] = None,
        num_perturbations: int = 3,
        perplex_thred: Optional[int] = None,
        **generation_kwargs,
    ):

        generator = load_generator()  # make it a step output?
        spacy_model = get_spacy_model("en_core_web_sm")  # TODO: for now.

        # TODO: make more efficient by flattening/unflattening and using batches for generation.
        all_sentences = []

        for idx, sentence in enumerate(processed_sentences):
            prompt_list = prompts[idx]  # list of str prompts

            generated = generate_and_clean_batch(
                prompts=prompts,
                generator=generator,
                n=num_perturbations,
                is_clean_verb_prefix=False,
                **generation_kwargs,
            )

            validated_set = []
            orig_doc = sentence.spacy_doc

            if generated:
                merged = [val for sublist in generated for val in sublist]
                # validate

                for raw_generated in merged:
                    try:
                        prompt_dict = parse_filled_prompt(
                            raw_generated, nlp=spacy_model, is_compute_vidx=True
                        )
                    except:
                        continue
                    generated_s = prompt_dict.sentence
                    if generated_s in validated_set or generated_s.lower() == orig_doc.text.lower():
                        continue
                    is_valid = True
                    # TODO:
                    # generated_doc = spacy_model(generated_s)
                    # if perplex_thred is not None:
                    #     eop = compute_edit_ops(orig_doc, generated_doc)
                    #     pp = self._compute_delta_perplexity(eop)
                    #     is_valid = pp.pr_sent < perplex_thred and pp.pr_phrase < perplex_thred
                    # if is_valid:
                    #     predicted = self.srl_predict(generated_s)
                    #     prompt_dict = add_predictions_to_prompt_dict(prompt_dict, predicted)
                    #     is_valid = is_vaild and is_followed_ctrl(prompt_dict, generated_doc, self.spacy_processor)
                    if is_valid:
                        validated_set.append(generated_s)

            all_sentences.append(validated_set)

        return all_sentences


# def generate_for_prompts(perturbed, generator, nlp,
#     base_doc=None, perplex_scorer=None, perplex_thred=20, **kwargs):
#     """ Gets generations for prompts
#     Args:
#         perturbed: list of Munch objects with attribute prompt
#     """
#     generated_sentences = get_generated_sentences(
#         generator, [pert.prompt for pert in perturbed], **kwargs)
#     assert len(perturbed) == len(generated_sentences)
#     new_perturbed = []
#     for orig_pert, generations in zip(perturbed, generated_sentences):
#         #assert len(generations) == 1
#         for gen in generations:
#             pert = deepcopy(orig_pert)
#             _, gen = extract_header_from_prompt(gen)
#             pert.generated = gen
#             # TODO debug nested exception handling: want to catch *BadGenerationError* but error handling currently broken
#             try: pert.clean_generated = parse_filled_prompt(gen, nlp=nlp)['sentence']
#             except: pert.clean_generated = None
#             if pert.clean_generated and \
#                 perplex_scorer and perplex_thred is not None \
#                 and base_doc and nlp:
#                 doc = nlp(pert.clean_generated)
#                 # edit operations to identify changed phrases
#                 eops = compute_edit_ops(base_doc, doc)
#                 # perplexity score
#                 pp = compute_delta_perplexity(eops, perplex_scorer)
#                 if pp.pr_sent > perplex_thred or pp.pr_phrase > perplex_thred:
#                     pert.clean_generated = None
#             new_perturbed.append(pert)
#     return new_perturbed

# def is_bad_generation(gen): return "sanatate" in gen
