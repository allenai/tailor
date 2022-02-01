from munch import Munch
from tailor.common.latest_utils import (
    get_verb_tense,
    get_verb_voice,
    get_keyword_specificity,
    RANDOM_TAG,
)


def is_followed_ctrl(generated_dict, doc=None, nlp=None):
    verifies = compute_ctrl_stats_per_prompt(generated_dict, doc, nlp)
    if not verifies:
        print("Compute ctrl stats failed!!")
        return False
    is_follow = True
    for v in verifies:
        if v.is_verb:
            is_follow = is_follow and v.correct_content and v.correct_tense and v.correct_voice
        else:
            is_follow = is_follow and v.correct_content and v.correct_tag and v.correct_specifity
    return is_follow


def compute_ctrl_stats_per_prompt(generated_dict, doc=None, nlp=None):
    """Compute whether a generation follows its query meta.
    Args:
        generated_dict (Munch): output of `add_predictions_to_prompt_dict`
        doc (Doc, Optional): the generated_doc doc of the sentence.
            Defaults to None, but cannot be None together with nlp.
        nlp : spacy processor, for generating doc if doc is none

    Returns:
        dict[]: a list of
    """

    def is_match_keyword(keyword, span):
        if keyword == "*":
            return True
        return keyword.lower() in span.text.lower() or keyword.lower() in span.lemma_.lower()

    is_valid = generated_dict.is_valid
    # is_core = generated_dict.meta.is_core
    if doc is None and nlp is None:
        return []
    if doc is None:
        doc = nlp(generated_dict.sentence)
    meta = generated_dict.meta
    vidx = generated_dict.vidx
    # create a list of verification
    verifies = []
    verifies.append(
        Munch(
            is_verb=True,
            submeta=meta,
            correct_tense=is_valid and get_verb_tense(doc[vidx], doc) == meta.vtense,
            correct_voice=is_valid and get_verb_voice(doc[vidx]) == meta.vvoice,
            correct_content=is_valid and is_match_keyword(meta.vlemma, doc[vidx]),
        )
    )

    def verify(annotations, submeta):
        correct_tag = None
        correct_content = submeta.tlemma == RANDOM_TAG
        correct_specifity = correct_content or submeta.tlemma_type is None
        for ann in annotations:
            # find a match
            correct_tag = ann.pred == submeta.tag
            content = doc[ann.start : ann.end]
            if not correct_tag:
                continue
            if not correct_specifity:
                correct_specifity = submeta.tlemma_type == get_keyword_specificity(
                    submeta.tlemma, [c.text for c in content]
                )
            if not correct_content:
                correct_content = is_match_keyword(submeta.tlemma, content) and correct_specifity
            if correct_tag and correct_content:
                break
        return Munch(
            is_verb=False,
            submeta=submeta,
            correct_content=is_valid and correct_content,
            correct_specifity=is_valid and correct_specifity,
            correct_tag=is_valid and correct_tag,
        )

    for submeta in meta.core_args + meta.noncore_args:
        verifies.append(verify(generated_dict.annotations, submeta))
    return verifies
