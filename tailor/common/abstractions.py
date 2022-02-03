# from enum import Enum
from typing import Dict, List, NamedTuple, Optional

from munch import Munch

from tailor.common.utils import SpacyDoc


class ProcessedSentence(NamedTuple):
    """
    Abstraction for a sentence processed with spacy and srl tagger.

    sentence : :class:`str`
        The original sentence string.

    spacy_doc : :class:`SpacyDoc`
        The spacy doc for the sentence string.

    verbs : :class:`List[Dict]`
        The list of detected verbs in the sentence. Each verb `Dict`
        contains all the tags for that verb.
    """

    sentence: str
    spacy_doc: SpacyDoc
    verbs: List[Dict]  # Dict: {"verb": str, "tags": List[str]}

    def get_tags_list(self) -> List[List[str]]:
        return [verb_dict["tags"] for verb_dict in self.verbs]


class PromptObject(NamedTuple):
    """
    TODO
    """

    prompt: Optional[str] = None
    answer: Optional[str] = None
    meta: Optional[Munch] = None  # TODO: use a PromptMeta abstraction.
    name: Optional[str] = None
    description: Optional[str] = None


def _munch_to_prompt_object(
    prompt_munch: Munch, name: Optional[str] = None, description: Optional[str] = None
):
    return PromptObject(
        prompt=prompt_munch.prompt,
        answer=prompt_munch.answer,
        meta=prompt_munch.meta,
        name=name,
        description=description,
    )


class GeneratedPrompt(NamedTuple):

    """
    The input string, repeated.
    """

    prompt_no_header: str

    """
    The natural language sentence
    """
    sentence: str

    """
    The meta info of the control, see output to `extract_meta_from_prompt`
    """
    meta: Munch  # TODO: PromptMeta abstraction.

    """
    The identified spans being changed, and the verb.
        [{tag: 'VERB', star: 6, end: 7}, {tag: 'FILL1', start: 0, end: 1}]
    """
    annotations: List[Munch]

    """
    The tokenized words of sentence
    """
    words: List[str]

    """
    The verb index
    """
    vidx: int

    """
    Name of the perturbation applied
    """
    name: Optional[str] = None

    """
    User-level description
    """
    description: Optional[str] = None

    """
    Is the generation valid
    """
    is_valid: Optional[bool] = None

    """
    Perplexities of generation
    """
    perplexities: Optional[Munch] = None


# class Specificities(Enum):
#     COMPLETE: str = "complete"
#     PARTIAL: str = "partial"
#     SPARSE: str = "sparse"


# class VerbVoice(Enum):
#     ACTIVE: str = "active"
#     PASSIVE: str = "passive"


# class VerbTense(Enum):
#     PRESENT: str = "present"
#     FUTURE: str = "future"
#     PAST: str = "past"


# class NonCoreArgs(NamedTuple):
#     tlemma: str
#     tlemma_type: Optional[str]  # TODO: confirm
#     raw_tag: str
#     tag: str  # TODO: add checks for correctness
#     blank_idx: List[int]  # TODO: should this be Tuple[int, int]?


# class PromptMeta(NamedTuple):

#     noncore_args: List[NonCoreArgs]

#     blank_indexes: List[List[int]]  # TODO:  are indexes always lists of 2? should they be a tuple?

#     answers: List[str]

#     agent: str

#     patient: str

#     frameset_id: str

#     vvoice: VerbVoice  # ACTIVE/PASSIVE

#     vtense: VerbTense  # PRESENT/FUTURE/PAST

#     vlemma: str

#     doc: SpacyDoc
