from copy import deepcopy
import itertools
from munch import Munch
from typing import Callable, List, NamedTuple, Optional, Union
from tango.common.registrable import Registrable
from tailor.common.latest_utils import (
    get_core_idxes_from_meta,
    capitalize_by_voice,
    get_keyword_candidates_for_span,
    parse_keyword_type,
    convert_tag2readable,
    get_arg_span,
)


class Perturbation(NamedTuple):

    perturb_str: str
    perturb_meta: Optional[Munch] = (None,)
    name: Optional[str] = None


class PerturbFunction(Registrable):
    def __call__(self, *args, **kwargs):  # TODO: maybe fix args?
        raise NotImplementedError


class PerturbStringFunction(Registrable):
    def __call__(self, prompt_meta, *args, **kwargs) -> Union[Perturbation, List[Perturbation]]:
        raise NotImplementedError


@PerturbStringFunction.register("change_voice")
class ChangeVoice(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> Union[Perturbation, List[Perturbation]]:

        vtense = prompt_meta.vtense
        target_voice = "active" if prompt_meta.vvoice == "passive" else "passive"

        perturb_str = (
            f"CONTEXT(DELETE_TEXT),VERB(CHANGE_TENSE({vtense}),CHANGE_VOICE({target_voice}))"
        )
        return Perturbation(perturb_str=perturb_str, perturb_meta=prompt_meta, name="change_voice")


@PerturbStringFunction.register("change_tense")
class ChangeTense(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> Union[Perturbation, List[Perturbation]]:
        perturb_str = "VERB(CHANGE_TENSE())"
        return Perturbation(perturb_str=perturb_str, perturb_meta=prompt_meta, name="change_tense")


@PerturbStringFunction.register("change_lemma")
class ChangeLemma(PerturbStringFunction):
    def __call__(self, prompt_meta, lemma: str, *args, **kwargs) -> Union[Perturbation, List[Perturbation]]:  # type: ignore
        perturb_str = f"VERB(CHANGE_LEMMA({lemma}))"
        return Perturbation(perturb_str=perturb_str, perturb_meta=prompt_meta, name="change_lemma")


@PerturbStringFunction.register("delete_text")
class DeleteText(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> Union[Perturbation, List[Perturbation]]:
        perturb_str = "CONTEXT(DELETE_TEXT)"
        return Perturbation(perturb_str=perturb_str, perturb_meta=prompt_meta, name="delete_text")


@PerturbStringFunction.register("delete_punctuation")
class DeletePunctuation(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> Union[Perturbation, List[Perturbation]]:
        perturb_str = "CONTEXT(DELETE_PUNCT)"
        return Perturbation(
            perturb_str=perturb_str, perturb_meta=prompt_meta, name="delete_punctuation"
        )


@PerturbStringFunction.register("swap_core_with_context")
class SwapCoreWithContext(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> Union[Perturbation, List[Perturbation]]:
        perturb_str = "CORE(SWAP_CORE)"
        return Perturbation(
            perturb_str=perturb_str, perturb_meta=prompt_meta, name="swap_core_with_context"
        )


@PerturbStringFunction.register("swap_core_without_context")
class SwapCoreWithoutContext(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> Union[Perturbation, List[Perturbation]]:
        perturb_str = "CONTEXT(DELETE_TEXT),CORE(SWAP_CORE)"
        return Perturbation(
            perturb_str=perturb_str, perturb_meta=prompt_meta, name="swap_core_without_context"
        )


def _filter_keywords_default(keywords):
    return [keywords.pop()[1]]


def replace_keyword_with_phenomenon(
    prompt_meta: Munch,
    core_arg_to_change: str,
    other_argument: str,
    phenomenon_from_other_argument: str,
    filter_keywords: Callable = _filter_keywords_default,  # change to registrable.
    specificity: str = "complete",
    do_capitalize_by_voice: bool = True,
    perturb_name: Optional[str] = None,
):
    perturbations = []
    if other_argument:
        assert phenomenon_from_other_argument is not None
        arg_span = get_arg_span(prompt_meta, other_argument)
        keyword_origin = other_argument  # May be required.
        if arg_span is not None:
            keyword_type = parse_keyword_type(phenomenon_from_other_argument)
            keywords = get_keyword_candidates_for_span(arg_span, keyword_type)

            keywords = filter_keywords(keywords)
            prompt_meta = deepcopy(prompt_meta)
            for keyword in keywords:
                if do_capitalize_by_voice:
                    core_idx = get_core_idxes_from_meta(prompt_meta)
                    agent = (
                        None
                        if core_idx.aidx is None
                        else prompt_meta.core_args[core_idx.aidx].tlemma
                    )
                    patient = (
                        None
                        if core_idx.pidx is None
                        else prompt_meta.core_args[core_idx.pidx].tlemma
                    )
                    if core_arg_to_change == "AGENT":
                        keyword, patient = capitalize_by_voice(prompt_meta.vvoice, keyword, patient)
                        if patient is not None:
                            prompt_meta.core_args[core_idx.pidx].tlemma = patient
                    elif core_arg_to_change == "PATIENT":
                        agent, keyword = capitalize_by_voice(prompt_meta.vvoice, agent, keyword)
                        if agent is not None:
                            prompt_meta.core_args[core_idx.aidx].tlemma = agent

                perturb_str = f"CONTEXT(DELETE_TEXT),NONCORE(ALL:DELETE),CORE({core_arg_to_change}:CHANGE_CONTENT({keyword}),CHANGE_SPECIFICITY(complete))"
                perturbations.append(
                    Perturbation(
                        perturb_str=perturb_str, perturb_meta=prompt_meta, name=perturb_name
                    )
                )

    return perturbations


@PerturbStringFunction.register("shorten_core_argument")
class ShortenCoreArgument(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> Union[Perturbation, List[Perturbation]]:

        perturbations = []

        args_to_shorten = ["AGENT", "PATIENT"]
        for core_arg_to_change in args_to_shorten:
            core_idx = get_core_idxes_from_meta(prompt_meta)
            if (
                core_arg_to_change == "AGENT"
                and core_idx.aidx is None
                or core_arg_to_change == "PATIENT"
                and core_idx.pidx is None
            ):
                return None

            perturbations += replace_keyword_with_phenomenon(
                prompt_meta,
                core_arg_to_change,
                other_argument=core_arg_to_change,
                phenomenon_from_other_argument="ROOT",
            )

        return perturbations
