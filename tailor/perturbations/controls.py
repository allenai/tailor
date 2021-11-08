"""
Define perturbation controls
"""
import logger
from typing import NamedTuple, Optional
import warnings
import re
from munch import Munch

from ..common.abstractions import Prompt
from .tag_utils import *


def format_warning(msg, *args, **kwargs):
    return str(msg) + "\n"


warnings.formatwarning = format_warning

logging.captureWarnings(True)

"""
TODO: docstrings for ALL.
"""
WRAPPERS = ["VERB", "CORE", "NONCORE", "CONTEXT"]
CONTEXT_CHANGES = ["CHANGE_IDX", "DELETE_TEXT", "DELETE_PUNCT"]
VFORM_ONLY = ["CHANGE_LEMMA", "CHANGE_TENSE", "CHANGE_VOICE"]
CORE_ONLY = ["SWAP_CORE"]
CORE_AND_NONCORE = ["CHANGE_CONTENT", "CHANGE_SPECIFICITY", "CHANGE_TAG"]
NONCORE_ONLY = ["DELETE"]  # TODO: add deletion capabilities to agent/patient too?

SPECIFICITIES = ["sparse", "partial", "complete"]
VERB_TENSES = ["present", "future", "past"]


CONTROL_CODES = Munch(
    {
        k: k
        for k in WRAPPERS
        + CONTEXT_CHANGES
        + VFORM_ONLY
        + CORE_ONLY
        + CORE_AND_NONCORE
        + NONCORE_ONLY
    }
)


class Perturbations(NamedTuple):
    """
    TODO: named tuple for now. may expand to full class with functionality;
    how to apply them: logic in head_prompt_utils.gen_prompt_by_perturb_meta
    """

    def get_prompt(self, *args, **kwargs) -> Prompt:
        pass
        # raise NotImplementedError


class ContextPerturbations(Perturbations):

    is_change_idx: bool = False
    is_delete_text: bool = False
    is_delete_punct: bool = False

    def get_prompt(self, *args, **kwargs):
        # if change index, randomly shuffle
        # TODO: make this change more meaningful (target idx?)
        if context_changes.is_change_idx:
            # TODO: Sherry changed this; The function didn't have empty_only arg?
            # new_meta = shuffle_empty_blank_indexes(new_meta, empty_only=False)
            # if None, shuffle
            if context_changes.idx_changes is None:
                new_meta = shuffle_empty_blank_indexes(new_meta)
            else:
                new_blank_appearance_indexes = new_meta.blank_appearance_indexes
                for idx, appearance in enumerate(new_blank_appearance_indexes):
                    len_doc = len(new_meta.doc)
                    new_idx_changes = {}
                    for orig, target in context_changes.idx_changes.items():
                        # negative value means relative idx; convert
                        if orig < 0:
                            orig += len_doc
                        if target < 0:
                            target += len_doc
                        new_idx_changes.update({orig: target})
                    context_changes.idx_changes = new_idx_changes
                    if appearance in context_changes.idx_changes:
                        target_idx = context_changes.idx_changes[appearance]
                        diff = target_idx - appearance
                        new_blank_appearance_indexes[idx] = appearance + diff
                new_meta.blank_appearance_indexes = new_blank_appearance_indexes

        new_prompt, new_answer = gen_prompt_and_answer_by_meta(
            new_meta, return_sequence=return_sequence
        )

        # TODO: update meta info and answer; right now, hackily deleting words in prompt
        if context_changes.is_delete_text or context_changes.is_delete_punct:
            header, nonheader = extract_header_from_prompt(new_prompt)
            words = nonheader.split(" ")
            new_words = []
            for word in words:
                word = word.strip()
                if word.startswith("<extra_id_"):
                    new_words.append(word)
                    continue
                # TODO: make more robust; now, if is_delete_text, delete all words that are not in string punctuation
                if context_changes.is_delete_punct and word in string.punctuation:
                    continue
                if context_changes.is_delete_text and not word in string.punctuation:
                    continue
                new_words.append(word)
            new_nonheader = " ".join(new_words)
            new_prompt = header + " " + new_nonheader


class VerbPerturbations(Perturbations):

    is_change_vtense: bool = False
    is_change_vvoice: bool = False
    is_change_vlemma: bool = False
    target_tense: bool = False
    target_voice: bool = False
    target_vlemma: bool = False


class TagChanges(Perturbations):

    is_delete: bool = False
    is_change_tag: bool = False
    target_tag: Optional[str] = None
    is_change_content: bool = False
    target_content: Optional[str] = None
    is_change_content_type: bool = False
    target_content_type: bool = False


class CorePerturbations(Perturbations):

    is_swap_core: bool = False
    is_change_agent: bool = False
    is_change_patient: bool = False
    agent_changes: Optional[TagChanges] = None
    patient_changes: Optional[TagChanges] = None


class NonCorePerturbations(Perturbations):
    is_change_noncore: bool = (
        False  # TODO: Is this required now that we have split out into separate class?
    )
    changes_by_arg: Optional[Dict[str, TagChanges]] = None


def parse_wrappers(control_code_str: str, wrapper: str) -> str:
    r_search = re.search(rf"{wrapper}\((?P<wrapper_changes>[^;]+)\)", control_code_str)
    wrapper_input = None if not r_search else r_search.group("wrapper_changes")
    return wrapper_input


def parse_change_context(control_code_str: str) -> ContextPerturbations:
    # TODO: build functionality to set target blank position
    """Helper function parse changes to context from string perturbation code
    Parses whole change_type_str to get context specific changes

    Note: Context specific changes must be wrapped in CONTEXT tag

    Example:
        input: "CONTEXT(DELETE_TEXT,CHANGE_IDX)"
        output: Munch({
                    'idx_changes': None,
                    'is_change_idx': True,
                    'is_delete_text': True,
                    'is_delete_punct': False,
                    })

    idx_changes is a dict mapping original indices to new indices for blank tokens
        if idx_changes is not supplied and is_change_idx, will randomly shuffle empty (?) blank indices at perturbation
    Args:
        control_code_str (str): the meta ctrl code.

    Returns:
        Munch object (example above)

    # TODO: update docstring.
    """

    context_change_str = parse_wrappers(control_code_str, CONTROL_CODES.CONTEXT)

    # no changes to context
    if context_change_str is None:
        if (
            CONTROL_CODES.DELETE_TEXT in change_type_str
            or CONTROL_CODES.DELETE_PUNCT in change_type_str
            or CONTROL_CODES.CHANGE_IDX in change_type_str
        ):
            warnings_message = (
                f"Context change wrapper ({CONTROL_CODES.CONTEXT}) not found in supplied perturb string "
                + f"({control_code_str}) but found parts of {[CONTROL_CODES.CHANGE_IDX, CONTROL_CODES.DELETE_TEXT, CONTROL_CODES.DELETE_PUNCT]} in string. "
                + f"Did you mean to wrap these changes in context wrapper {CONTROL_CODES.CONTEXT}? "
                + "\n"
                + f"Default parsing behavior: Returning no context changes. "
            )
            warnings.warn(warnings_message)

        return ContextPerturbations(
            is_delete_text=False,
            is_delete_punct=False,
            is_change_idx=False,
        )

    is_change_idx = CHANGE_IDX in context_change_str
    is_delete_text = DELETE_TEXT in context_change_str
    is_delete_punct = DELETE_PUNCT in context_change_str

    def parse_idx_changes(idx_changes):
        changes_by_idx = {}
        r = re.findall(r"(?P<subchange>[^\|\),]+)", idx_changes)
        for match in r:
            changes = match.split(":")
            assert len(changes) == 2
            orig, end = changes
            orig, end = int(orig), int(end)
            changes_by_idx.update({int(orig): int(end)})
        return changes_by_idx

    r = re.search(rf"{CONTROL_CODES.CHANGE_IDX}\((?P<idx_changes>[^\)]+)\)", context_change_str)
    if is_change_idx:
        if r:
            idx_changes = r.group("idx_changes")
            idx_changes = parse_idx_changes(idx_changes)
        else:
            idx_changes = None

    context_changes = ContextPerturbations(
        idx_changes=idx_changes,
        is_change_idx=is_change_idx,
        is_delete_text=is_delete_text,
        is_delete_punct=is_delete_punct,
    )
    return context_changes


def parse_change_verb(control_code_str: str) -> VerbPerturbations:
    """Helper function get target vform from string perturbation code
    Parses whole control_code_str to get verb specific changes

    Note: Verb specific changes must be wrapped in VERB tag

    Example:
        input: "VERB(CHANGE_TENSE(past),CHANGE_VOICE(),CHANGE_LEMMA(buy))"
        output: Munch({
                    'is_change_vtense': True,
                    'is_change_vvoice': True,
                    'is_change_vlemma': True,
                    'target_tense': 'past',
                    'target_voice': None,
                    'target_vlemma': 'buy'
                    })

    Args:
        control_code_str (str): the meta ctrl code.

    Returns:
        Munch object (example above)

    """

    verb_change_str = parse_wrappers(control_code_str, CONTROL_CODES.VERB)

    # no changes to verb
    if verb_change_str is None:
        if (
            (CONTROL_CODES.CHANGE_LEMMA in control_code_str)
            or (CONTROL_CODES.CHANGE_TENSE in control_code_str)
            or (CONTROL_CODES.CHANGE_VOICE in control_code_str)
        ):
            warnings_message = (
                f"Verb change wrapper ({CONTROL_CODES.VERB}) not found in supplied perturb string "
                + f"({control_code_str}) but found parts of {[CONTROL_CODES.CHANGE_LEMMA, CONTROL_CODES.CHANGE_TENSE, CONTROL_CODES.CHANGE_VOICE]} in string. "
                + f"Did you mean to wrap these changes in verb wrapper {CONTROL_CODES.VERB}? "
                + "\n"
                + f"Default parsing behavior: Returning no verb changes. "
            )
            warnings.warn(warnings_message)

        return VerbPerturbations(
            is_change_vtense=False,
            is_change_vvoice=False,
            is_change_vlemma=False,
            target_tense=None,
            target_voice=None,
            target_vlemma=None,
        )

    is_change_vtense = CONTROL_CODES.CHANGE_TENSE in verb_change_str
    is_change_vvoice = CONTROL_CODES.CHANGE_VOICE in verb_change_str
    is_change_vlemma = CONTROL_CODES.CHANGE_LEMMA in verb_change_str

    r = re.search(rf"{CONTROL_CODES.CHANGE_TENSE}\((?P<vtense>[^,\)]+)\)", verb_change_str)
    vtense = None if not r else r.group("vtense")
    if not vtense:
        tense = None
    else:
        tense = vtense
        if vtense not in ["present", "future", "past"]:
            raise ValueError(f"Incorrect form for verb tense {vtense}")

    r = re.search(rf"{CONTROL_CODES.CHANGE_VOICE}\((?P<vvoice>[^,\)]+)\)", verb_change_str)
    vvoice = None if not r else r.group("vvoice")
    if not vvoice:
        voice = None
    else:
        voice = vvoice
        if vvoice not in ["active", "passive"]:
            raise ValueError(f"Incorrect form for verb voice {vvoice}")

    r = re.search(rf"{CONTROL_CODES.CHANGE_LEMMA}\((?P<target_vlemma>[^\)]+)\)", verb_change_str)
    target_vlemma = None if not r else r.group("target_vlemma")

    verb_changes = VerbPerturbations(
        is_change_vtense=is_change_vtense,
        is_change_vvoice=is_change_vvoice,
        is_change_vlemma=is_change_vlemma,
        target_tense=tense,
        target_voice=voice,
        target_vlemma=target_vlemma,
    )
    return verb_changes


def parse_change_noncore(change_type_str):
    """Helper function to get noncore changes from string perturbation code
    Parses whole change_type_str to get noncore specific changes

    Notes:
        - Noncore changes must be wrapped in NONCORE tag
        - Changes can either be specificied per-tag (with format TAG: *insert tag-specific-changes*)
              or for all tags (with format ALL: *insert changes here*)


    Example:
        input: "NONCORE(TEMPORAL:CHANGE_CONTENT(place),
                                 CHANGE_TAG(LOCATIVE),
                                 CHANGE_SPECIFICITY(complete)
                        |MODAL:CHANGE_CONTENT)"
        output: Munch({
                    'is_change_noncore': True,
                    'changes_by_arg': {'TEMPORAL': Munch({
                                                        'is_delete': False,
                                                        'is_change_tag': True,
                                                        'is_change_content': True,
                                                        'target_content': 'place',
                                                        'is_change_content_type': True,
                                                        'target_tag': 'LOCATIVE',
                                                        'target_content_type': 'complete'}),
                                       'MODAL': Munch({'is_delete': False,
                                                       'is_change_tag': False,
                                                       'is_change_content': True,
                                                       'target_content': None,
                                                       'is_change_content_type': False,
                                                       'target_tag': None,
                                                       'target_content_type': None})}})

        input: "NONCORE(ALL:CHANGE_TAG)"
        output: Munch({
                    'is_change_noncore': True,
                    'changes_by_arg': {'ALL': Munch({
                                                'is_delete': False,
                                                'is_change_tag': True,
                                                'is_change_content': False,
                                                'target_content': None,
                                                'is_change_content_type': False,
                                                'target_tag': None,
                                                'target_content_type': None})}})

    Args:
        change_type_str (str): the meta ctrl code.

    Returns:
        Munch object specifying change type
            Has two main attributes:
                is_change_noncore: bool
                changes_by_arg: dict(str, Munch object)
                    key=tag, value=Munch object specifying changes (see example for attributes)
    """

    def parse_sub_change_type(sub_change_type):
        """Parses change type per argument"""
        is_change_tag = CHANGE_TAG in sub_change_type
        is_change_content = CHANGE_CONTENT in sub_change_type
        is_delete = DELETE in sub_change_type
        is_change_content_type = CHANGE_SPECIFICITY in sub_change_type
        r = re.search(rf"{CHANGE_CONTENT}\((?P<target_content>[^\)]+)\)*", sub_change_type)
        if is_change_content and r:
            target_content = r.group("target_content")
        else:
            target_content = None

        if is_delete and any([is_change_content, is_change_tag, is_change_content_type]):
            warnings_message = (
                f"Found delete code {DELETE} in supplied perturb string with additional "
                + f"perturbation types for sub perturb string '{sub_change_type}'. "
                + "Tag will be deleted and additional perturbations will be ignored. "
            )
            warnings.warn(warnings_message)

        r = re.search(rf"{CHANGE_SPECIFICITY}\((?P<target_content_type>[^\)]+)\)*", sub_change_type)
        if is_change_content_type and r:
            target_content_type = r.group("target_content_type")
            assert target_content_type in [
                "sparse",
                "complete",
                "partial",
            ], "Unrecognized content type, must be one of sparse/complete/partial"
        else:
            target_content_type = None

        r = re.search(rf"{CHANGE_TAG}\((?P<target_tag>[^\)]+)\)*", sub_change_type)
        if is_change_tag and r:
            target_tag = r.group("target_tag")
        else:
            target_tag = None

        return Munch(
            {
                "is_delete": is_delete,
                "is_change_tag": is_change_tag,
                "is_change_content": is_change_content,
                "target_content": target_content,
                "is_change_content_type": is_change_content_type,
                "target_tag": target_tag,
                "target_content_type": target_content_type,
            }
        )

    is_change_noncore = NONCORE in change_type_str
    changes_by_arg = {}

    r = re.search(rf"{NONCORE}\((?P<noncore_change>[^;]+)\)", change_type_str)
    noncore_change_str = None if not r else r.group("noncore_change")
    if not noncore_change_str:
        if any([tag in change_type_str for tag in get_argm_values()]):
            warnings_message = (
                f"Noncore change wrapper ({NONCORE}) not found in supplied perturb string "
                + f"({change_type_str}) but found specific tags in string. "
                + f"Did you mean to wrap these changes in noncore wrapper {NONCORE}? "
                + "\n"
                + f"Default parsing behavior: Returning no noncore changes. "
            )
            warnings.warn(warnings_message)
    else:
        # Try matching ALL to see if changing all noncore args
        r = re.match(r"ALL:(?P<sub_change_type>[^\|\)]+)", noncore_change_str)
        if r:
            sub_change_type = r.group("sub_change_type")
            parsed = parse_sub_change_type(sub_change_type)
            changes_by_arg = {"ALL": parsed}
        else:
            r = re.findall(r"(?P<tag>[^\,\|)\(]+):(?P<sub_change_type>[^\|]+)", noncore_change_str)
            for tag, sub_change_type in r:
                parsed = parse_sub_change_type(sub_change_type)
                changes_by_arg[tag] = parsed

    return Munch(is_change_noncore=is_change_noncore, changes_by_arg=changes_by_arg)


def parse_change_core(change_type_str):
    """Helper function to get core changes from string perturbation code
    Parses whole change_type_str to get core specific changes

    Notes:
        - Core changes must be wrapped in CORE tag
        - Changes must be specified for AGENT and PATIENT separately
            (with format AGENT/PATIENT: *insert changes here*)


    Example:
        input: "CORE(AGENT:CHANGE_SPECIFICITY,CHANGE_CONTENT(Mr)
                    |PATIENT:CHANGE_CONTENT,CHANGE_SPECIFICITY(partial))"
        output: Munch({
                    'is_swap_core': False,
                    'is_change_agent': True,
                    'is_change_patient': True,
                    'agent_changes': Munch({
                                        'is_delete': False,
                                        'is_change_content': True,
                                        'target_content': 'Mr',
                                        'is_change_content_type': True,
                                        'target_content_type': None}),
                    'patient_changes': Munch({
                                        'is_delete': False,
                                        'is_change_content': True,
                                        'target_content': None,
                                        'is_change_content_type': True,
                                        'target_content_type': 'partial'})})

    Args:
        change_type_str (str): the meta ctrl code.

    Returns:
        Munch object specifying change type
            Has attributes:
                is_swap_core: bool (whether to swap agent/patient tags)
                is_change_agent: bool
                is_change_patient: bool
                agent_changes: Munch object with attributes:
                    is_change_content: bool
                    target_content: str, optional
                    is_change_content_type: bool (i.e. whether to change specificity)
                    target_content_type: str, optional
                patient_changes: Munch object (has same attributes as agent_changes)
    """
    # TODO: allow changing both AGENT/CORE at once (like in parse_change_noncore() with "ALL" tag)

    def parse_sub_change_type(sub_change_type):
        is_delete = DELETE in sub_change_type
        is_change_content = CHANGE_CONTENT in sub_change_type
        is_change_content_type = CHANGE_SPECIFICITY in sub_change_type
        r = re.search(rf"{CHANGE_CONTENT}\((?P<target_content>[^\)]+)\)", sub_change_type)
        if is_change_content and r:
            target_content = r.group("target_content")
        else:
            target_content = None

        if is_delete and any([is_change_content, is_change_content_type]):
            warnings_message = (
                f"Found delete code {DELETE} in supplied perturb string with additional "
                + f"perturbation types for sub perturb string '{sub_change_type}'. "
                + "Tag will be deleted and additional perturbations will be ignored. "
            )
            warnings.warn(warnings_message)

        r = re.search(rf"{CHANGE_SPECIFICITY}\((?P<target_content_type>[^\)]+)\)", sub_change_type)
        if is_change_content_type and r:
            target_content_type = r.group("target_content_type")
            assert target_content_type in [
                "sparse",
                "complete",
                "partial",
            ], "Unrecognized content type, must be one of sparse/complete/partial"
        else:
            target_content_type = None

        return Munch(
            {
                "is_delete": is_delete,
                "is_change_content": is_change_content,
                "target_content": target_content,
                "is_change_content_type": is_change_content_type,
                "target_content_type": target_content_type,
            }
        )

    r = re.search(rf"{CORE}\((?P<core_changes>[^;]+)\)", change_type_str)
    core_change_str = None if not r else r.group("core_changes")

    dummy_change = Munch(
        is_change_content=False,
        target_content=None,
        is_change_content_type=False,
        target_content_type=None,
    )

    if core_change_str is None:
        if "AGENT" in change_type_str or "PATIENT" in change_type_str:
            warnings_message = (
                f"Core change wrapper ({CORE}) not found in supplied perturb string "
                + f"({change_type_str}) but found AGENT/PATIENT in string. "
                + f"Did you mean to wrap these changes in core wrapper {CORE}? "
                + "\n"
                + f"Default parsing behavior: Returning no core changes. "
            )
            warnings.warn(warnings_message)
        return Munch(
            is_swap_core=False,
            is_change_agent=False,
            is_change_patient=False,
            agent_changes=dummy_change,
            patient_changes=dummy_change,
        )
    else:
        patient_r = re.search(r"PATIENT:(?P<sub_change_type>([^\|]*|$))", core_change_str)
        agent_r = re.search(r"AGENT:(?P<sub_change_type>([^\|]*|$))", core_change_str)
        if agent_r:
            sub_change_type = agent_r.group("sub_change_type")
            parsed = parse_sub_change_type(sub_change_type)
            agent_changes = parsed
            is_change_agent = True
        else:
            agent_changes = dummy_change
            is_change_agent = False

        if patient_r:
            sub_change_type = patient_r.group("sub_change_type")
            parsed = parse_sub_change_type(sub_change_type)
            patient_changes = parsed
            is_change_patient = True
        else:
            patient_changes = dummy_change
            is_change_patient = False

        is_swap_core = SWAP_CORE in core_change_str

        return Munch(
            is_swap_core=is_swap_core,
            is_change_agent=is_change_agent,
            is_change_patient=is_change_patient,
            agent_changes=agent_changes,
            patient_changes=patient_changes,
        )


# TODO: implement more sanity checks to make sure change_type_str takes the correct form,
# and if not, raise ValueError (in addition to existing warnings)

# TODO: add capabilities to only change *one* aspect of verb form and keep original
def parse_change_type_meta(change_type_str):
    """A parser that translate string control codes to perturb meta
    So that at generation time it's easier to input the desired change.

    Notes:
        - Calls functions:
                parse_change_context,
                parse_change_core,
                parse_change_noncore,
                parse_change_verb
            to parse context, core, noncore, and verb changes respectively.
            See these functions for more information.
        - Make sure wrappers are separated by semicolons for correct parsing behavior.

    Args:
        change_type_str (str): the meta ctrl code.
            Comma separated combination of:
                - CONTEXT(*context-specific-changes*)
                - VERB(*verb-specific-changes*)
                - CORE(*core-specific-changes*)
                - NONCORE(*noncore-specific-changes*)

    Example:
        input: "CONTEXT(CHANGE_IDX(3:4|0:-1));VERB(CHANGE_VFORM);CORE(SWAP_CORE);NONCORE(ALL:CHANGE_TAG)"
        output: Munch({
                    'context_changes': Munch({
                                        'idx_changes': {3: 4, 0: -1},
                                        'is_delete_text': False,
                                        'is_delete_punct': False,
                                        'is_change_idx': True}),
                    'core_changes': Munch({
                                        'is_swap_core': True,
                                        'is_change_agent': False,
                                        'is_change_patient': False,
                                        'agent_changes': Munch({
                                                            'is_change_content': False,
                                                            'target_content': None,
                                                            'is_change_content_type': False,
                                                            'target_content_type': None}),
                                        'patient_changes': Munch({
                                                            'is_change_content': False,
                                                            'target_content': None,
                                                            'is_change_content_type': False,
                                                            'target_content_type': None})}),
                    'noncore_changes': Munch({
                                        'is_change_noncore': True,
                                        'changes_by_arg': {
                                                'ALL': Munch({
                                                        'is_change_tag': True,
                                                        'is_change_content': False,
                                                        'target_content': None,
                                                        'is_change_content_type': False,
                                                        'target_tag': None,
                                                        'target_content_type': None})}}),
                    verb_changes': Munch({
                                        'is_change_vform': True,
                                        'is_change_vlemma': False,
                                        'target_tense': None,
                                        'target_voice': None,
                                        'target_vlemma': None})})
    Raises:
        ValueError: If the type cannot be parsed

    Returns:
        Munch: {
            context_changes: Munch object (see output of parse_change_context)
            core_changes: Munch object (see output of parse_change_core)
            noncore_changes: Munch object (see output of parse_change_noncore)
            verb_changes: Munch object (see output of parse_change_verb)
        }
    """
    perturb_meta = None

    context_changes = parse_change_context(change_type_str)
    verb_changes = parse_change_verb(change_type_str)
    noncore_changes = parse_change_noncore(change_type_str)
    core_changes = parse_change_core(change_type_str)

    # TODO: this does not handle CORE and ADVERBIAL right; gives warning when it shouldn't
    present_wrappers = re.findall(rf"[^N]*{CORE}|{NONCORE}|{VERB}|{CONTEXT}", change_type_str)
    num_semicolons = change_type_str.count(";")
    # there should be 1 less semicolon than wrappers
    if len(present_wrappers) != num_semicolons + 1:
        warnings_message = (
            f"Warning: Found multiple wrappers in change_type_str '{change_type_str}' "
            + "but they do not seem to be separated properly, with semicolons ';'."
            + f"Found {len(present_wrappers)} wrappers and {num_semicolons} semicolons."
            + "This may lead to incorrect parsing behavior."
        )
        warnings.warn(warnings_message)

    perturb_meta = Munch(
        context_changes=context_changes,
        core_changes=core_changes,
        noncore_changes=noncore_changes,
        verb_changes=verb_changes,
    )

    return perturb_meta
