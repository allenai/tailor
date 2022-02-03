from typing import List
import itertools
from copy import deepcopy

from tailor.common.utils import SpacyDoc
from tailor.common.abstractions import PromptObject, _munch_to_prompt_object
from tailor.common.perturb_function import PerturbFunction
from tailor.common.utils.head_prompt_utils import (
    capitalize_by_voice,
    convert_tag2readable,
    gen_prompt_by_perturb_str,
    get_arg_span,
    get_core_idxes_from_meta,
    get_keyword_candidates_for_span,
    is_equal_headers,
    parse_keyword_type,
)


@PerturbFunction.register("replace-core-with-subsequence")
class ReplaceCoreWithSubsequence(PerturbFunction):
    def __call__(
        self,
        spacy_doc: SpacyDoc,
        intermediate_prompt: PromptObject,
        tags: List[List[str]],
        args_to_blank: List[List[str]],
        *args,
        **kwargs,
    ):

        new_prompts = []
        prompt = deepcopy(intermediate_prompt)
        new_keywords_by_arg = {"AGENT": set(), "PATIENT": set()}
        keyword_origins = {}
        # for each core arg, sample replacement args and store in new_keywords_by_arg
        for arg_to_replace in ["AGENT", "PATIENT"]:
            args_to_consider = [
                convert_tag2readable(prompt.meta.vlemma, t, None) for t in args_to_blank
            ]
            args_to_consider = [t for t in args_to_consider if t not in ["VERB"]]
            for arg in args_to_consider:
                arg_span = get_arg_span(prompt.meta, arg)
                if arg_span is None:
                    continue
                keywords = [
                    kw[1]
                    for kw in get_keyword_candidates_for_span(
                        arg_span, parse_keyword_type("NOUN_CHUNKS,UNCASED")
                    )
                    if kw[1] != ""
                ]
                for keyword in keywords:
                    keyword_origins[keyword] = arg
                new_keywords_by_arg[arg_to_replace].update(set(keywords))

        for agent_keyword, patient_keyword in itertools.product(
            new_keywords_by_arg["AGENT"], new_keywords_by_arg["PATIENT"]
        ):
            # if replacing both core args with original keywords from those spans, skip
            if (
                keyword_origins[agent_keyword] == "AGENT"
                and keyword_origins[patient_keyword] == "PATIENT"
            ):
                continue
            if agent_keyword == patient_keyword:
                continue
            # equivalent to swap core
            core_idx = get_core_idxes_from_meta(prompt.meta)
            if core_idx.pidx is None or core_idx.aidx is None:
                continue
            if (
                agent_keyword == prompt.meta.core_args[core_idx.aidx].tlemma
                and patient_keyword == prompt.meta.core_args[core_idx.pidx].tlemma
            ):
                continue
            # fix cases of keywords to encourage generating full sentence
            agent_keyword, patient_keyword = capitalize_by_voice(
                prompt.meta.vvoice, agent_keyword, patient_keyword
            )
            # TODO: do we want to delete context here?
            # change both agent/patient keywords
            perturb_str = (
                "CONTEXT(DELETE_TEXT);NONCORE(ALL:DELETE);"
                f"CORE(AGENT:CHANGE_CONTENT({agent_keyword}),CHANGE_SPECIFICITY(complete)"
                f"PATIENT:CHANGE_CONTENT({patient_keyword}),CHANGE_SPECIFICITY(complete))"
            )
            perturbed = gen_prompt_by_perturb_str(spacy_doc, tags, perturb_str, prompt.meta)

            if perturbed is None:
                continue

            if not is_equal_headers(perturbed.prompt, prompt.prompt):

                perturbed_object = _munch_to_prompt_object(
                    perturbed, name="replace_core_with_subsequence", description="changes_meaning"
                )
                new_prompts.append(perturbed_object)
            return new_prompts
