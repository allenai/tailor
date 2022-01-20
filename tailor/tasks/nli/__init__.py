import re
from tailor.common.perturb_function import PerturbFunction
from tailor.common.tag_utils import DEFAULT_FRAME_SET_PATH
from tailor.common.old_utils import *
from tailor.common.perturbation_controls import *

# from tailor.common.latest_utils import get_unique_prompts


def is_equal_headers(p1, p2):
    """Helper function check for equality of headers between two prompts
    Useful for making sure that edited prompts are different
    Args:
        p1 (str): prompt
        p2 (str): prompt
    Returns:
        bool: Whether the two prompts have equal headers
    """
    return extract_header_from_prompt(p1)[0] == extract_header_from_prompt(p2)[0]


def get_unique_prompts(prompts):
    """Helper function to get unique prompts given list of prompts
    Helpful when we care about a looser notion of equality than exact string equality
    Calls is_equal_prompts() to check equality of prompts
    """
    prompt_set = []
    for p in prompts:
        if not any(is_equal_prompts(p, exist_p) for exist_p in prompt_set):
            prompt_set.append(p)
    return prompt_set


def is_equal_prompts(p1, p2):
    """Helper function check for equality of two prompts
    Insensitive to differences in space and punctuation in context
    Useful for making sure that edited prompts are different
    Args:
        p1 (str): prompt
        p2 (str): prompt
    Returns:
        bool: Whether the two prompts have equal prompts
    """

    def remove_punctuation(s):
        return re.sub(r"[.!?,-]", "", s)

    p1_head, p1_context = extract_header_from_prompt(p1)
    p2_head, p2_context = extract_header_from_prompt(p2)
    p1_context = remove_punctuation(p1_context).replace(" ", "").strip()
    p2_context = remove_punctuation(p2_context).replace(" ", "").strip()
    return p1_head.strip() == p2_head.strip() and p1_context.strip() == p2_context.strip()


def get_arg_span(meta, short_tag):
    """Helper function to get doc span of a given arg
    Args:
        meta: Munch object
        short_tag: readable tag of arg whose span we want
    """
    original_blank_idx = None
    args = meta.core_args if short_tag in ["AGENT", "PATIENT"] else meta.noncore_args
    original_blank_tuple = [arg.blank_idx for arg in args if arg.tag == short_tag]
    try:
        original_blank_idx = meta.blank_indexes.index(original_blank_tuple[0])
    except:
        original_blank_idx = None
    # can only shorten an existing argument
    if original_blank_idx is None:
        return None

    blank_idx = meta.blank_indexes[original_blank_idx]
    return meta.doc[blank_idx[0] : blank_idx[1]]


def capitalize_by_voice(verb_voice, agent_kw, patient_kw):
    """Helper function to fix capitalizations of agent/patient keywords based on verb voice
    This is to encourage generating full sentence without hallucinating context, since generator is case sensitive.
    Only capitalizes/lowercases if kws are not None

    Args:
        verb_voice (str): active/passive
        agent_kw (str, or None): agent keyword
        patient_kw (str, or None): patient keyword
    Returns:
        agent_kw (str, or None): new agent keyword
        patient_kw (str, or None): new patient keyword
    """
    if agent_kw is None and patient_kw is None:
        warnings_message = "Trying to change capitalization of agent and patient keywords, but got None for both arguments"
        warnings.warn(warnings_message)
    if verb_voice == "passive":
        if agent_kw is not None:
            agent_kw = lowercase_keyword(agent_kw)
        if patient_kw is not None:
            patient_kw = uppercase_keyword(patient_kw)
    else:
        if agent_kw is not None:
            agent_kw = uppercase_keyword(agent_kw)
        if patient_kw is not None:
            patient_kw = lowercase_keyword(patient_kw)
    return agent_kw, patient_kw


@PerturbFunction.register("change_voice")
class ChangeVoice(PerturbFunction):
    """
    Meaning-Preserving Strategy.
    Creates prompts that change voice of prompt by changing verb voice and keywords.
    Note: Deletes context that is not part of predicate-argument structure of that verb
        i.e., we get the following perturbations for sentence:
            The athlete who was seen by the judges yesterday called the manager
            -> The judges saw the athlete yesterday.
            -> The manager was called by the athlete who was seen by their judges yeterday.
        NOT the following two (which change the voice of "seen" while leaving the context "yesterday called the manager):
            \-> The judges saw the athlete who yesterday called the manager.
            \-> The athlete who the judges saw yesterday called the manager.
    Example:
        sentence: The athlete who was seen by the judges yesterday called the manager.
        intermediate prompt:
            [VERB+active+past: call | AGENT+complete: The athlete who was seen by the judges yesterday | PATIENT+complete: the manager]
            <extra_id_0> <extra_id_1> <extra_id_2> <extra_id_3> <extra_id_4> <extra_id_5> <extra_id_6> <extra_id_7><extra_id_8><extra_id_9>
        edited prompt (returned):
            [VERB+passive+past: call | AGENT+complete: by the athlete who was seen by the judges yesterday | PATIENT+complete: The manager]
            <extra_id_0> <extra_id_1> <extra_id_2> <extra_id_3> <extra_id_4> <extra_id_5> <extra_id_6> <extra_id_7> <extra_id_8> <extra_id_9>
                -> example generation (not part of prompt):
                [PATIENT: The manager] is [VERB: called] [AGENT : by the athlete who was seen by. the judges yesterday]
    Args:
        sentence (str): sentence from which to extract relative clause
        pred (dict): prediction object; output of calling predict() on AllenNLP SRL Predictor
        doc (Doc): spacy doc for sentence

    Returns:
        list[str]: list of prompts
    """

    # intermediate: List[prompts] # One per verb.

    def __call__(self, processed, intermediate, frameset_path=DEFAULT_FRAME_SET_PATH):
        doc = processed.spacy_doc
        srl_tag_list = processed.get_tags_list()
        prompts = []
        for idx, prompt in enumerate(intermediate):  # per verb.
            tags = srl_tag_list[idx]

            core_idx = get_core_idxes_from_meta(prompt.meta)
            if core_idx.pidx is None or core_idx.aidx is None:
                continue
            vtense = prompt.meta.vtense
            # upper case to encourage generating full sentence; need to capitalize keywords by *target* voice
            target_voice = "active" if prompt.meta.vvoice is "passive" else "passive"
            (
                prompt.meta.core_args[core_idx.aidx].tlemma,
                prompt.meta.core_args[core_idx.pidx].tlemma,
            ) = capitalize_by_voice(
                target_voice,
                prompt.meta.core_args[core_idx.aidx].tlemma,
                prompt.meta.core_args[core_idx.pidx].tlemma,
            )
            perturb_str = (
                f"CONTEXT(DELETE_TEXT),VERB(CHANGE_TENSE({vtense}),CHANGE_VOICE({target_voice}))"
            )
            perturbed = gen_prompt_by_perturb_str(
                doc, tags, perturb_str, prompt.meta, frameset_path=frameset_path
            )
            if perturbed is None:
                continue
            if not is_equal_headers(perturbed.prompt, prompt.prompt):
                prompts.append(perturbed.prompt)
        return get_unique_prompts(prompts)


@PerturbFunction.register("change_voice_single")
class ChangeVoiceSingle(PerturbFunction):
    """
    Meaning-Preserving Strategy.
    Creates prompts that change voice of prompt by changing verb voice and keywords.
    Note: Deletes context that is not part of predicate-argument structure of that verb
        i.e., we get the following perturbations for sentence:
            The athlete who was seen by the judges yesterday called the manager
            -> The judges saw the athlete yesterday.
            -> The manager was called by the athlete who was seen by their judges yeterday.
        NOT the following two (which change the voice of "seen" while leaving the context "yesterday called the manager):
            \-> The judges saw the athlete who yesterday called the manager.
            \-> The athlete who the judges saw yesterday called the manager.
    Example:
        sentence: The athlete who was seen by the judges yesterday called the manager.
        intermediate prompt:
            [VERB+active+past: call | AGENT+complete: The athlete who was seen by the judges yesterday | PATIENT+complete: the manager]
            <extra_id_0> <extra_id_1> <extra_id_2> <extra_id_3> <extra_id_4> <extra_id_5> <extra_id_6> <extra_id_7><extra_id_8><extra_id_9>
        edited prompt (returned):
            [VERB+passive+past: call | AGENT+complete: by the athlete who was seen by the judges yesterday | PATIENT+complete: The manager]
            <extra_id_0> <extra_id_1> <extra_id_2> <extra_id_3> <extra_id_4> <extra_id_5> <extra_id_6> <extra_id_7> <extra_id_8> <extra_id_9>
                -> example generation (not part of prompt):
                [PATIENT: The manager] is [VERB: called] [AGENT : by the athlete who was seen by. the judges yesterday]
    Args:
        sentence (str): sentence from which to extract relative clause
        pred (dict): prediction object; output of calling predict() on AllenNLP SRL Predictor
        doc (Doc): spacy doc for sentence

    Returns:
        list[str]: list of prompts
    """

    # intermediate: List[prompts] # One per verb.

    def __call__(self, spacy_doc, intermediate_prompt, tags, frameset_path=DEFAULT_FRAME_SET_PATH):

        prompt = intermediate_prompt  # TODO: is deepcopy necessary?

        core_idx = get_core_idxes_from_meta(prompt.meta)
        if core_idx.pidx is None or core_idx.aidx is None:
            return None
        vtense = prompt.meta.vtense
        # upper case to encourage generating full sentence; need to capitalize keywords by *target* voice
        target_voice = "active" if prompt.meta.vvoice is "passive" else "passive"
        (
            prompt.meta.core_args[core_idx.aidx].tlemma,
            prompt.meta.core_args[core_idx.pidx].tlemma,
        ) = capitalize_by_voice(
            target_voice,
            prompt.meta.core_args[core_idx.aidx].tlemma,
            prompt.meta.core_args[core_idx.pidx].tlemma,
        )
        perturb_str = (
            f"CONTEXT(DELETE_TEXT),VERB(CHANGE_TENSE({vtense}),CHANGE_VOICE({target_voice}))"
        )
        perturbed = gen_prompt_by_perturb_str(
            spacy_doc, tags, perturb_str, prompt.meta, frameset_path=frameset_path
        )
        if perturbed is None:
            return None
        if not is_equal_headers(perturbed.prompt, prompt.prompt):
            return perturbed.prompt
