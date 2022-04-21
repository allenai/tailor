from copy import deepcopy
from typing import Callable, List, NamedTuple, Optional, Union

from munch import Munch
from tango.common.registrable import Registrable

from tailor.common.abstractions import PromptObject
from tailor.common.utils import SpacyDoc
from tailor.common.utils.head_prompt_utils import (
    capitalize_by_voice,
    get_arg_span,
    get_core_idxes_from_meta,
    get_keyword_candidates_for_span,
    parse_keyword_type,
)


class Perturbation(NamedTuple):

    perturb_str: str
    perturb_meta: Optional[Munch] = None
    name: Optional[str] = None

    # This is for generalizing things like "preserves_meaning"
    description: Optional[str] = None  # TODO (Alexis): should this be a Munch for more flexibility?


class PerturbFunction(Registrable):
    def __call__(
        self,
        spacy_doc: SpacyDoc,
        intermediate_prompt: PromptObject,
        tags: List[List[str]],
        args_to_blank: List[List[str]],
        *args,
        **kwargs,
    ) -> Union[PromptObject, List[PromptObject]]:
        raise NotImplementedError


class PerturbStringFunction(Registrable):
    def __call__(
        self, prompt_meta, *args, description=None, **kwargs
    ) -> Union[Perturbation, List[Perturbation]]:
        raise NotImplementedError


@PerturbStringFunction.register("change_voice")
class ChangeVoice(PerturbStringFunction):
    def __call__(
        self, prompt_meta, *args, description=None, **kwargs
    ) -> Union[Perturbation, List[Perturbation]]:

        target_voice = "active" if prompt_meta.vvoice == "passive" else "passive"

        perturb_str = (
            f"CONTEXT(DELETE_TEXT);VERB(CHANGE_VOICE({target_voice}))"
        )
        return Perturbation(
            perturb_str=perturb_str,
            perturb_meta=prompt_meta,
            name="change_voice",
            description=description,
        )


@PerturbStringFunction.register("change_tense")
class ChangeTense(PerturbStringFunction):
    def __call__(
        self, prompt_meta, *args, description=None, **kwargs
    ) -> Union[Perturbation, List[Perturbation]]:
        perturb_str = "VERB(CHANGE_TENSE())"
        return Perturbation(
            perturb_str=perturb_str,
            perturb_meta=prompt_meta,
            name="change_tense",
            description=description,
        )


@PerturbStringFunction.register("change_lemma")
class ChangeLemma(PerturbStringFunction):
    def __call__(  # type: ignore
        self, prompt_meta, lemma: str, *args, description=None, **kwargs
    ) -> Union[Perturbation, List[Perturbation]]:
        perturb_str = f"VERB(CHANGE_LEMMA({lemma}))"
        return Perturbation(
            perturb_str=perturb_str,
            perturb_meta=prompt_meta,
            name="change_lemma",
            description=description,
        )


@PerturbStringFunction.register("delete_text")
class DeleteText(PerturbStringFunction):
    def __call__(
        self, prompt_meta, *args, description=None, **kwargs
    ) -> Union[Perturbation, List[Perturbation]]:
        perturb_str = "CONTEXT(DELETE_TEXT)"
        return Perturbation(
            perturb_str=perturb_str,
            perturb_meta=prompt_meta,
            name="delete_text",
            description=description,
        )


@PerturbStringFunction.register("delete_punctuation")
class DeletePunctuation(PerturbStringFunction):
    def __call__(
        self, prompt_meta, *args, description=None, **kwargs
    ) -> Union[Perturbation, List[Perturbation]]:
        perturb_str = "CONTEXT(DELETE_PUNCT)"
        return Perturbation(
            perturb_str=perturb_str,
            perturb_meta=prompt_meta,
            name="delete_punctuation",
            description=description,
        )


@PerturbStringFunction.register("swap_core_with_context")
class SwapCoreWithContext(PerturbStringFunction):
    def __call__(
        self, prompt_meta, *args, description=None, **kwargs,
    ) -> Union[Perturbation, List[Perturbation]]:
        perturb_str = "CORE(SWAP_CORE)"
      
        # Capitalize the keywords by voice to improve generation quality 
        core_idx = get_core_idxes_from_meta(prompt_meta)
        if (core_idx.pidx is not None and core_idx.aidx is not None): 
            # give in opposite order because want to reverse the capitalization logic:
            # i.e., patient becomes new agent, so uppercase current patient (rather than agent) if verb voice is active 
            patient, agent = capitalize_by_voice(
                                prompt_meta.vvoice, 
                                prompt_meta.core_args[core_idx.pidx].tlemma,
                                prompt_meta.core_args[core_idx.aidx].tlemma) 
            prompt_meta.core_args[core_idx.pidx].tlemma = patient
            prompt_meta.core_args[core_idx.aidx].tlemma = agent 

        return Perturbation(
            perturb_str=perturb_str,
            perturb_meta=prompt_meta,
            name="swap_core_with_context",
            description=description,
        )


@PerturbStringFunction.register("swap_core_without_context")
class SwapCoreWithoutContext(PerturbStringFunction):
    def __call__(
        self, prompt_meta, *args, description=None, **kwargs
    ) -> Union[Perturbation, List[Perturbation]]:
        perturb_str = "CONTEXT(DELETE_TEXT);CORE(SWAP_CORE)"
        
        # Capitalize the keywords by voice to improve generation quality 
        core_idx = get_core_idxes_from_meta(prompt_meta)
        if (core_idx.pidx is not None and core_idx.aidx is not None): 
            # give in opposite order because want to reverse the capitalization logic:
            # i.e., patient becomes new agent, so uppercase current patient (rather than agent) if verb voice is active 
            patient, agent = capitalize_by_voice(
                                prompt_meta.vvoice, 
                                prompt_meta.core_args[core_idx.pidx].tlemma,
                                prompt_meta.core_args[core_idx.aidx].tlemma) 
            prompt_meta.core_args[core_idx.pidx].tlemma = patient
            prompt_meta.core_args[core_idx.aidx].tlemma = agent 
       
        return Perturbation(
            perturb_str=perturb_str,
            perturb_meta=prompt_meta,
            name="swap_core_without_context",
            description=description,
        )


def _filter_keywords_default(keywords):
    return [keywords.pop()[1]]


def replace_keyword_with_phenomenon(
    prompt_meta: Munch,
    core_arg_to_change: str,
    other_argument: str,
    phenomenon_from_other_argument: str,
    filter_keywords: Callable = _filter_keywords_default,  # TODO: change to registrable?.
    specificity: str = "complete",
    do_capitalize_by_voice: bool = True,
    perturb_name: Optional[str] = None,
    description: Optional[str] = None,
):
    perturbations = []
    if other_argument:
        assert phenomenon_from_other_argument is not None
        arg_span = get_arg_span(prompt_meta, other_argument)
        # keyword_origin = other_argument  # May be required.
        if arg_span is not None:
            keyword_type = parse_keyword_type(phenomenon_from_other_argument)
            keywords = get_keyword_candidates_for_span(arg_span, keyword_type)

            keywords = filter_keywords(keywords)
            prompt_meta = deepcopy(prompt_meta)
            for keyword in keywords:
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

                perturb_str = (
                    "CONTEXT(DELETE_TEXT);NONCORE(ALL:DELETE);"
                    f"CORE({core_arg_to_change}:CHANGE_CONTENT({keyword}),CHANGE_SPECIFICITY(complete))"
                )
                perturbations.append(
                    Perturbation(
                        perturb_str=perturb_str,
                        perturb_meta=prompt_meta,
                        name=perturb_name,
                        description=description,
                    )
                )

    return perturbations


@PerturbStringFunction.register("shorten_core_argument")
class ShortenCoreArgument(PerturbStringFunction):
    def __call__(
        self, prompt_meta, *args, description=None, **kwargs
    ) -> Union[Perturbation, List[Perturbation]]:

        perturbations: List[Perturbation] = []

        args_to_shorten = ["AGENT", "PATIENT"]
        for core_arg_to_change in args_to_shorten:
            core_idx = get_core_idxes_from_meta(prompt_meta)
            if (
                core_arg_to_change == "AGENT"
                and core_idx.aidx is None
                or core_arg_to_change == "PATIENT"
                and core_idx.pidx is None
            ):
                return perturbations

            perturbations += replace_keyword_with_phenomenon(
                prompt_meta,
                core_arg_to_change,
                other_argument=core_arg_to_change,
                phenomenon_from_other_argument="ROOT",
                perturb_name="shorten_core_argument",
                description=description,
            )

        return perturbations
