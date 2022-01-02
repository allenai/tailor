from enum import Enum
from typing import List, NamedTuple, Optional
from munch import Munch

from tailor.common.util import SpacyDoc


class Specificities(Enum):
    COMPLETE: str = "complete"
    PARTIAL: str = "partial"
    SPARSE: str = "sparse"


class VerbVoice(Enum):
    ACTIVE: str = "active"
    PASSIVE: str = "passive"


class VerbTense(Enum):
    PRESENT: str = "present"
    FUTURE: str = "future"
    PAST: str = "past"


class NonCoreArgs(NamedTuple):
    tlemma: str
    tlemma_type: Optional[str]  # TODO: confirm
    raw_tag: str
    tag: str  # TODO: add checks for correctness
    blank_idx: List[int]  # TODO: should this be Tuple[int, int]?


class PromptMeta(NamedTuple):

    noncore_args: List[NonCoreArgs]

    blank_indexes: List[List[int]]  # TODO:  are indexes always lists of 2? should they be a tuple?

    answers: List[str]

    agent: str

    patient: str

    frameset_id: str

    vvoice: VerbVoice  # ACTIVE/PASSIVE

    vtense: VerbTense  # PRESENT/FUTURE/PAST

    vlemma: str

    doc: SpacyDoc


class Prompt(NamedTuple):  # TODO: add more functionality?

    prompt: str

    meta: PromptMeta

    answer: str  # TODO: what's this?


class CriteriaForPerturbation:
    """
    TODO: add details
    A lot of the perturbation strategies might have criteria that must be met
    for the perturbation to be applied, and these criteria are task-specific,
    so we’ll want the user to supply them. A given sentence will have multiple
    prompts (one for each predicate), and we often only want to apply the perturbations
    for specific predicates and if some argument(s) contain some linguistic phenomenon
    (eg. prepositional phrases, a particular verb voice, etc.)
    """

    pass
