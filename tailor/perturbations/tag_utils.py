"""
Helpers for identifying tags from span, and making them readable
"""
import difflib
from typing import Dict, List

# TODO: how to handle B-R-ARGs? right now being encoded in header separately

TAG2READABLE_MAPPING: Dict[str, str] = {
    "COM": "COMITATIVE",
    "LOC": "LOCATIVE",
    "DIR": "DIRECTIONAL",
    "GOL": "GOAL",
    "MNR": "MANNER",
    "TMP": "TEMPORAL",
    "EXT": "EXTENT",
    "REC": "RECIPROCAL",
    "PRD": "PREDICATE",
    "PRP": "PURPOSE",
    "CAU": "CAUSE",
    "DIS": "DISCOURSE",
    "ADV": "ADVERBIAL",
    "ADJ": "ADVEJECTIVAL",
    "MOD": "MODAL",
    "NEG": "NEGATION",
    "DSP": "SPEECH",
    "LVB": "LIGHT",
    "CXN": "CONSTRUCTION",
}

CORE_TAGS: List[str] = ["VERB", "AGENT", "PATIENT"]

READABLE2TAG_MAPPING: Dict[str, str] = {v: k for k, v in TAG2READABLE_MAPPING.items()}
ADDITIONAL_CASES: List[str] = ["instrument", "attribute", "start", "end"]


def get_argm_values():
    return list(TAG2READABLE_MAPPING.values()) + [cc.upper() for cc in ADDITIONAL_CASES]


def get_argm_and_core_values():
    return get_argm_values() + CORE_TAGS


GOLD_TAGS: List[str] = get_argm_and_core_values()


def find_most_likely_tag(raw_tag: str, gold_tags: List[str] = GOLD_TAGS) -> str:
    """Helper function for finding the most likely tag if the generated
    tag is not exactly the gold ones.

    Args:
        raw_tag (str): The imperfect generated tag
        gold_tags (List[str]): The golds, or output from get_argm_values.

    Returns:
        [str]: the matched tag.
    """
    raw_tag = raw_tag.upper().strip()
    if raw_tag in gold_tags:
        return raw_tag
    scores = [(gt, difflib.SequenceMatcher(a=raw_tag, b=gt).quick_ratio()) for gt in gold_tags]
    scores = sorted(scores, key=lambda s: -s[1])
    # print(scores)
    return scores[0][0]
