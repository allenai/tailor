"""
general design doc can be seen in:
https://docs.google.com/document/d/1HbhO9EAFlXpYA3_rSdrXlBHCz4aynWNUTm63dlQDAlc/edit?usp=sharing
"""

import itertools
import os
import re
import string
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from copy import deepcopy
from random import random

import more_itertools as mit
import numpy as np
from munch import Munch
from nltk.corpus import stopwords
from openie import StanfordOpenIE

from tailor.common.utils.perturbation_controls import parse_change_type_meta
from tailor.common.utils.generate_utils import clean_punct 
from tailor.common.utils.tag_utils import (
    ADDITIONAL_CASES,
    DEFAULT_FRAME_SET_PATH,
    READABLE2TAG_MAPPING,
    TAG2READABLE_MAPPING,
    find_most_likely_tag,
    get_argm_and_core_values,
)


def format_warning(msg, *args, **kwargs):
    return str(msg) + "\n"


warnings.formatwarning = format_warning  # type: ignore


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


def convert_tag2readable(
    vlemma, raw_tag, frameset_id, role_dicts=None, frameset_path=DEFAULT_FRAME_SET_PATH
):
    """map a tag to readable text
    Prioritize returning mapping associated with *particular frameset id*
    Args:
        vlemma (str): the vlemma text
        tag (str): the raw tag to be mapped to human readable ones;
            Can be BIO format or just MOD/ARG0/etc.
        frameset_id (str): frameset id of verb.
            Used for getting appropriate raw tag -> short tag mapping
        role_dicts Optional(dict of dicts): role dicts for all frameset ids of the verb
            Defaults to None
            If not provided, will get role_dicts by calling get_possible_tags_by_vlemma()
            Can be provided to speed up this call.
        frameset_path (str, optional):
            the path to frameset. Defaults to "../label_contrast/srl/data/propbank-frames/frames/".
            cd ic/generators/
            git clone git@github.com:propbank/propbank-frames.git
    Returns:
        str: the human readable argument
    """
    clean_tag = raw_tag.split("-")[-1]
    if clean_tag == "V":
        return "VERB"
    if clean_tag == "ARG0":
        return "AGENT"
    if clean_tag == "ARG1":
        return "PATIENT"

    if f"{clean_tag}" in TAG2READABLE_MAPPING:
        return TAG2READABLE_MAPPING[f"{clean_tag}"]
    elif clean_tag in get_argm_and_core_values() + ["VERB"]:
        return clean_tag
    if role_dicts is None:
        role_dicts = get_possible_tags_by_vlemma(vlemma, frameset_path=frameset_path)
    if role_dicts is not None:
        if frameset_id in role_dicts.keys():
            frame_role_dict = role_dicts[frameset_id]
            if clean_tag in frame_role_dict:
                return frame_role_dict[clean_tag]
        # iterate over all possible frameset ids to find a readable tag
        for _, role_dict in role_dicts.items():
            if clean_tag in role_dict:
                return role_dict[clean_tag]
    # happens bc prop banks has changed since ontonotes was created?
    # i.e. leave.01 no longer exists; instead, it is leave.11
    return None


def get_tag_for_span(
    raw_tags,
    frameset_id,
    start,
    end,
    vlemma=None,
    is_convert_readable=False,
    frameset_path=DEFAULT_FRAME_SET_PATH,
):
    """Given the raw SRL tag, the start and end idx of the target span, retrive the
    key SRL label for the span. Can convert to readable format if also given the
    vlemma, which is needed for querying xml.
    Args:
        raw_tags (str[]): per-token tags in BIO format.
        frameset_id (str): frameset id of verb.
            Used for getting appropriate raw tag -> short tag mapping
        start (int): start index of span
        end (int): end intext of span
        vlemma (text, optional): the lemma for of the corresponding V. Defaults to None.
        is_convert_readable (bool, optional): If convert to readable label,
            e.g. MOD->MODAL. Defaults to False. Needs vlemma
        frameset_path (str, optional): the path to frameset. See `convert_tag2readable`
    Returns:
        str: the span. If does not exist, return None
    """
    start = start if start is not None and start > 0 else 0
    end = end if end is not None and start <= len(raw_tags) else len(raw_tags)
    local_tags = [clean_prefix_for_one_tag(t) for t in raw_tags[start:end]]
    if local_tags:
        label = None
        raw_tag = Counter(local_tags).most_common()[0][0]
        if is_convert_readable and raw_tag and vlemma:
            label = convert_tag2readable(vlemma, raw_tag, frameset_id, frameset_path=frameset_path)
        return label if label else raw_tag
    return None


def get_possible_tags_by_vlemma(vlemma, frameset_path=DEFAULT_FRAME_SET_PATH):
    """Get a collection of acceptable modifier tags given a vlemma.
    This can be used to create possible prompts at generation time.
    Args:
        vlemma (text, optional): the verb lemma.
        frameset_path (str, optional): the path to frameset.
            See `convert_tag2readable`
            Defaults to DEFAULT_FRAME_SET_PATH.
    Raises:
        ValueError: If cannot find verb in xml, raise an error
    Returns:
        meta dict of dicts, with frameset ids as meta keys: {
            [raw tag]: [readable tag]
        }
        Each dict corresponds to one roleset.
    """
    role_dicts = {}
    assert os.path.exists(frameset_path), f"Can't find frameset path: {frameset_path}"
    try:
        tree = ET.parse(os.path.join(frameset_path, f"{vlemma}.xml"))
        root = tree.getroot()
    except IOError:
        return None
    #        raise ValueError(f"Cannot find the verb: {vlemma}")
    for roleset in root.findall("predicate/roleset"):
        frameset_id = str(roleset.attrib["id"][-2:])
        role_dict = {}
        for role in roleset.findall("roles/role"):
            key = role.attrib["f"]
            #            if role.attrib['n'] == "0" or role.attrib['n'] == "1":
            #                continue
            arg = f"ARG{role.attrib['n']}"
            desc = role.attrib["descr"].split(" ")[0].split(",")[0].split(",")[0]
            # func = role.attrib["f"].split(" ")[0]
            #            if func == "ppt":
            #                role_dict[arg] = "PATIENT"
            #            elif func == "pag":
            #                role_dict[arg] = "AGENT"
            if desc in ADDITIONAL_CASES:
                role_dict[arg] = desc.upper()
            elif f"{key.upper()}" in TAG2READABLE_MAPPING:
                role_dict[arg] = TAG2READABLE_MAPPING[f"{key.upper()}"]
            elif roleset.find("vnrole"):
                role_dict[arg] = roleset.find("vnrole").attrib["vntheta"].upper()
        additional_values = set(TAG2READABLE_MAPPING.values()) - set(role_dict.values())
        # add other cases
        for v in additional_values:
            role_dict[READABLE2TAG_MAPPING[v]] = v
        if role_dict:
            role_dicts[frameset_id] = role_dict
    return role_dicts


######################################################################
# convert meta and header control codes
######################################################################


def flatten_fillins(doc, indxes, appearances, is_return_text=False):
    """Use index and corresponding fill ins to create a modified string.
    Useful for creating blanks.
    Args:
        doc (Doc): spacy doc
        indxes ([int, int][]): the indexes where the text should be modified
        fillins_by_idxes (str[][]): a list of candidate fillins at each space.
        is_return_text (bool, optional): If true, return the first possible
        changed text. Otherwise return a list. Defaults to False.
    Returns:
        [type]: [description]
    """
    text_arr = []
    # prev = 0
    tok_idxes_to_blank = set()
    for (start, end) in indxes:
        # if equal, then empty blank
        if start == end:
            continue
        for temp_idx in range(start, end):
            tok_idxes_to_blank.add(temp_idx)
    id_idx = 0
    for tok_idx, tok in enumerate(doc):
        if tok_idx in appearances:
            blank_counts = len([x for x in appearances if x == tok_idx])
            for _ in range(blank_counts):
                fillin = convert_idx2blank(id_idx)
                text_arr.append(fillin + tok.whitespace_)
                id_idx += 1
        if tok_idx not in tok_idxes_to_blank:
            text_arr.append(tok.text + tok.whitespace_)

    if len(doc) in appearances:
        blank_counts = len([x for x in appearances if x == len(doc)])
        for _ in range(blank_counts):
            fillin = convert_idx2blank(id_idx)
            text_arr.append(fillin + " ")
            id_idx += 1

    otexts = "".join(text_arr)
    return otexts.strip()


def get_vindex_by_tags(tags):
    """Get the index of the verb, based on verb tag B-V

    Args:
        tags (List[str]): SRL tags

    Returns:
        int: the index
    """

    if "B-V" not in tags:
        return None
    return tags.index("B-V")


def clean_prefix_for_one_tag(tag):
    """delete I/B- prefix for the BIO tag
    Args:
        tag (str): SRL tag
    Returns:
        str: the SRL tag with I/B- removed
    """
    return re.split(r"(B|I)\-", tag)[-1]


def extract_header_from_prompt(prompt):
    header = re.search(r"\[[^\]]+\]", prompt)
    if header:
        header = header.group(0)
    else:
        raise RuntimeError("Header not found in prompt!")  # TODO: more specific error?
    non_header = prompt.replace(header, "")
    return header, non_header


def extract_meta_from_prompt(prompt, return_header=False):
    """Extract a meta dict for a given prompt
    Args:
        prompt (str): a prompt in the head training format
        return_header (bool): whether or not to return the header with meta info
    Returns:
        Munch: {
            vvoice: active|passive
            vlemma: keyword
            vtense: verb tense
            core_args: [
                tlemma: keyword
                tlemma_type: sparse|complete|partial
                tag: modifier tag in readable format
            ]
            noncore_args: *same format as core args*
            match: the entire tag string
        }
    """

    # TODO: is this function robust? No way of getting some meta information
    # from the prompt (like non core args specific info)
    # TODO: what's the frameset?
    meta = Munch()
    # find matches in header (don't want to match generated text)
    header, _ = extract_header_from_prompt(prompt)
    assert header is not None, "Could not extract header from this prompt {prompt}"
    r = re.search(
        r"\[VERB\+(?P<vvoice>active|passive)\+(?P<vtense>past|present|future): (?P<vlemma>[^\]\|]+)",
        header,
    )
    if not r:
        return None, None
    meta.match = str(r)
    meta.vlemma = r.group("vlemma").strip()
    meta.vvoice = r.group("vvoice").strip()
    meta.vtense = r.group("vtense").strip()
    meta.noncore_args = []
    meta.core_args = []
    args = re.findall(r"\| (?P<tag>[^\]\|]+): (?P<tlemma>[^\]\|]+)", header)
    for tag, tlemma in args:
        if "+" in tag:
            tag, tlemma_type = tag.split("+")
        else:
            tlemma_type = None
        tlemma = tlemma.strip()
        if tag == "VERB":
            continue
        arg = Munch(tlemma=tlemma, tlemma_type=tlemma_type, raw_tag=None, tag=tag, blank_idx=None)
        if tag == "AGENT" or tag == "PATIENT":
            meta.core_args.append(arg)
        else:
            meta.noncore_args.append(arg)
    # meta.noncore_args = noncore_args
    # meta.is_core = True if len(noncore_args) == 0 else False
    meta.core_args = sorted(meta.core_args, key=lambda c: 0 if c.tag == "AGENT" else 1)
    if return_header:
        return meta, header
    return meta


def gen_header_by_meta(meta):
    """Translate header control based on meta
    - Always stores as VERB, AGENT, PATIENT and then modifiers in undetermined order
    Args:
        meta (Munch): the meta. See output of `extract_meta_from_prompt`.
    Returns:
        str: the header
    """
    header = f"[VERB+{meta.vvoice}+{meta.vtense}: {meta.vlemma}"
    meta.core_args = sorted(meta.core_args, key=lambda c: 0 if c.tag == "AGENT" else 1)
    for arg in meta.core_args + meta.noncore_args:
        tlemma_type = (
            "" if (arg.tlemma_type is None or arg.tlemma == RANDOM_TAG) else ("+" + arg.tlemma_type)
        )
        header += f" | {arg.tag}{tlemma_type}: {arg.tlemma}"
    header += "]"
    return header


######################################################################
# translate doc into prompts
######################################################################

RANDOM_TAG = "*"
EMPTY_FILL = ""
MAX_NUM_BLANKS = 10


def convert_idx2blank(idx):
    """To make t5 blanks
    Args:
        idx (int): id of the current blank
    Returns:
        str: <extra_id_{idx}>
    """
    return f"<extra_id_{idx}>"


def extract_aux_blanks(verb):
    """Gets indexes of spans of helper verbs based on a spacy verb token.
    - Excludes modals, since these are not conjugated and can remain invariant
    Args:
        verb (Token): verb
    Returns:
        (list of [int, int]: indexes, list of str: text for spans)
    """
    doc = verb.doc
    # do not include modal verbs bc these have own tag
    verb_children = [c for c in verb.children]
    aux_indexes = [
        c.i for c in verb_children if "aux" in c.dep_ and c.pos_ == "AUX" and c.tag_ != "MD"
    ]
    consecutive_indexes = [list(group) for group in mit.consecutive_groups(aux_indexes)]
    if aux_indexes == []:
        return [], []
    blank_indexes = [[min(span), max(span) + 1] for span in consecutive_indexes]
    answers = [" ".join([doc[idx].text for idx in group]) for group in consecutive_indexes]
    return blank_indexes, answers


def get_verb_voice(verb):
    """Helper function to get verb form of verb
    Args:
        verb (Token): verb

    Returns:
        str: "active" or "passive"
    """

    all_children = [v.dep_ for v in list(verb.children)]
    return "passive" if any(["pass" in a or "agent" in a for a in all_children]) else "active"


def get_verb_tense(verb, doc):
    # TODO: this is hacky; refine--maybe work directly with the Penn Treebank tags
    # in the original conll data
    # TODO: is current future tense detection robust? i.e. only returns future if
    # "will" or "shall" is auxiliary modal for verb
    # TODO: is "shall" returned as a modal--confirm
    """Helper function to heuristically get verb tense of verb
        - If there are auxiliary modal verbs and modal is "will", return "future"
            TODO: is this robust?
        - If not and first auxiliary verb is "be", return tense
          (don't want tense of "have" auxiliary, bc this is past tense)
            TODO: is this robust?
        - If not then, get tense of main verb
    Need to include modals in aux indices since can influence tense:
        i.e. "will" corresponds to future tense.
        At generation time, want verb tense in header to influence which modals are generated
    Args:
        verb (Token): verb

    Returns:
        str: "past", "present", or "future"
    """
    verb_children = [c for c in verb.children]
    modal_indexes = [
        c.i for c in verb_children if "aux" in c.dep_ and c.pos_ == "AUX" and c.tag_ == "MD"
    ]
    if modal_indexes:
        idx = modal_indexes[0]
        if doc[idx].text in ["will", "shall"]:
            return "future"
    # non modal auxiliary indices; need to check separately bc don't want tense of modal "would"
    # TODO: is this robust?
    # (if helper verb is "be" aka "was going" -> past then return tense of helper verb)
    aux_indices = [
        c.i for c in verb_children if "aux" in c.dep_ and c.pos_ == "AUX" and c.tag_ != "MD"
    ]
    if aux_indices:
        idx = aux_indices[0]
        if doc[idx].lemma_ == "be" or doc[idx].lemma_ == "do":
            if "Past" in doc[idx].morph.get("Tense"):
                return "past"
            elif "Pres" in doc[idx].morph.get("Tense"):
                return "present"
    if "Past" in verb.morph.get("Tense"):
        return "past"
    elif "Future" in verb.morph.get("Tense"):
        return "future"
    return "present"


def openie_extract(text):
    with StanfordOpenIE() as client:
        return client.annotate(text)


def get_keyword_candidates_for_span(span, keyword_type):

    """Get all possible candidate keywords for a span
    Design doc is here:
    https://docs.google.com/document/d/10_nnl0Q23BFx9To8_qaHzTU7kmQUJrLCQJqkgYAjQMo/edit#
    Demo here:
    notebooks/keyword experiment
    If use openie, need this package: https://pypi.org/project/stanford-openie/
    Args:
        span (Span): the span for extracting keywords
        keyword_type = Munch object with attributes:
            is_include_root (bool, optional): whether to include root token of span. Defaults to True.
            is_include_openie (bool, optional): whether to get openie extracted text. Defaults to True.
            is_include_connect (bool, optional): whether to include the starting ADP. Defaults to True.
            is_include_noun_chunks (bool, optional): whether to include noun chunks. Defaults to True.
            is_include_random_subtrees (bool, optional): whether to include random subtrees.
                Defaults to True.
            is_include_exact (bool, optional): whether to include exact text as keyword.
                Defaults to True.
            is_include_prefix (bool, optional): whether to include the first n tokens of the argument,
                where n is randomly selected. Defaults to True.
                Useful for dependency-parsing perturbations,
                i.e. [PATIENT: some talks with commanders and officials] keyword is "some talks with"
    Returns:
        set(tuple(str, str)): A set of possible candidates in the form of tuples:
            (keyword_type, keyword) where keyword_type is one of: sparse/partial/complete
                - complete: keyword has same num of tokens as the span
                - partial: keyword missing at most 5 tokens from span
                - sparse: keyword missing more than 5 tokens in span
    """

    root = span.root
    doc = span.doc
    candidate_set = set()

    # TODO: sometimes adding text, sometimes adding lemma: should work for training,
    # but make more principled
    if len(span) == 0:
        print(span, span.doc)
        return candidate_set
    if keyword_type.is_include_noun_chunks:
        candidate_set.update(set([r.text for r in span.noun_chunks]))
    if keyword_type.is_include_openie:
        relations = openie_extract(doc.text)
        for r in relations:
            for key, val in r.items():
                if val in span.text:
                    candidate_set.add(val)
    if keyword_type.is_include_random_subtrees:
        # randomly select three random subroot and take their results
        random_root_spans = np.random.choice(span, min(3, len(span)), replace=False)
        for r in random_root_spans:
            start = min(max(span.start, r.left_edge.i), span.end - 1)
            end = max(min(span.end, r.right_edge.i + 1), span.start)
            curr_subtree = doc[start:end]
            # if (
            #     end > start
            #     and end-start < 5
            #     and not all([t.is_punct or t.is_stop for t in curr_subtree]):
            # )
            if (
                end > start
                and len(curr_subtree) < len(span)
                and not all([t.is_punct or t.is_stop for t in curr_subtree])
            ):
                candidate_set.add(curr_subtree.text)
    if keyword_type.is_include_connect:
        if root.pos_ == "ADP":
            candidate_set.add(root.text)
        if len(span) > 0 and span[0].pos_ == "ADP":
            candidate_set.add(span[0].lemma_)
    if keyword_type.is_include_root:
        if root.pos_ == "NOUN" and [n for n in span.noun_chunks if root in n]:
            candidate_set.update(set([[n for n in span.noun_chunks if root in n][0].text]))
        else:
            while root.pos_ == "ADP" and len(list(root.children)):
                root = list(root.children)[0]
            candidate_set.add(root.lemma_)
    if keyword_type.is_include_exact:
        candidate_set.add(span.text)
    if keyword_type.is_include_prefix:
        num_tokens = np.random.choice(range(1, len(span) + 1), 1)[0]
        candidate_set.add(" ".join([t.text for t in span[:num_tokens]]))
    if not candidate_set:
        return set([(None, RANDOM_TAG)])
    candidate_set_tuples = set()
    # TODO fix hack: "complete" if keyword has same number of tokens as span
    for c in candidate_set:
        specificity = get_keyword_specificity(c, [s.text for s in span])
        # remove trailing punctuation
        if c[-1] in string.punctuation:
            c = c[:-1]
        # uncase
        if keyword_type.is_uncased:
            c = c.lower()
        candidate_set_tuples.add((specificity, c))
    return candidate_set_tuples


def get_keyword_specificity(keyword, full_content):
    if type(keyword) == str:
        keyword = keyword.split(" ")
    if type(full_content) == str:
        full_content = keyword.split(" ")
    if len(keyword) == len(full_content):
        specificity = "complete"
    elif len(full_content) - len(keyword) <= 5:
        specificity = "partial"
    else:
        specificity = "sparse"
    return specificity


def get_keyword_for_span(span, keyword_str, n=1):
    """get the content keyword
    Args:
        span (Span): the span for extracting keywords
        keyword_str (str): string specifying types of keywords to include
            See parse_keyword_type and get_keyword_candidates_for_span for more information
        n (int): number of candidate spans

    Returns:
        str|list[str]: the key of a given span when n=1,
            or a list of candidates when n is not 1.
            n=1 case is for compatibility
    """
    keyword_type = parse_keyword_type(keyword_str)
    candidates = list(get_keyword_candidates_for_span(span, keyword_type))
    if n == 1:
        idx = np.random.choice(len(candidates))
        return candidates[idx]
    else:
        idxes = list(np.random.choice(len(candidates), min(len(candidates), n), replace=False))
        return [candidates[idx] for idx in idxes]


# TODO: is this function robust to when args_to_blank includes arguments not in the prompt?
def extract_exist_blanks(
    doc,
    raw_tags,
    args_to_blank,
    frameset_id,
    nblanks,
    is_extract_first,
    keyword_str="NOUN_CHUNKS,RANDOM_SUBTREES,EXACT,PREFIX",
    vlemma=None,
    start=None,
    end=None,
    p_overblank=0,
    frameset_path=DEFAULT_FRAME_SET_PATH,
):
    """A helper function that translates docs into tuple of
    (keywords, blank indexes, answers). Called in prompt creation
    for core/not core, to extract existing information.
    Args:
        doc (Doc): spacy doc
        raw_tags (str[]): per-token tags in BIO format.
        frameset_id (str): frameset id of verb.
            Used for getting appropriate raw tag -> short tag mapping
        nblanks (int): the number of placeholders to create.
            For core, it's almost always 3 because of the three info.
            For noncore, it can be >=1, and we will have another function
            later that fill in the empty blanks -- as a way of varying the
            input place. (`add_extra_blanks`)
        is_extract_first (bool): Mostly for noncore. When there's multiple
            chunks of one information, whether to only extract and blank one.
        keyword_str (str): string specifying types of keywords to include
            See parse_keyword_type and get_keyword_candidates_for_span for more information
        args_to_blank (str[]): the arguments to be blanked, in an ordered
            list. The keyword/blank idxes/answer list all correspond to
            this list.
            For core, it's usually [V, ARG0, ARG1]
            For noncore, it's a len=1 arr with the targeted tag.
        start (int, optional): Only extract info in a given span.
            Defaults to None.
        end (int, optional): Only extract info in a given span.
    Returns:
        (Tuple): A tuple of (keywords, blank indexes, answers, tags, short_tags). Their indexes
        correspond to each other. That is, answers[0] is the groundtruth fill in
        for idxes[0], whose extracted keyword is keywords[0]
        (
            set[]: content keywords list, always contain the RANDOM *
            [int, int][]: the blanked indexes. If an entry is unfilled,
                it'd be [-1, -1]
            answers: the groundtruth fill in for the corresponding blank
            tags: raw tags, i.e. "ARGM-CAU"
            short_tags: control codes, i.e. "CAUSE"
        )
    """
    words = [t.text for t in doc]
    # set placeholders
    blank_indexes = [[-1, -1] for o in range(nblanks)]
    tags = [EMPTY_FILL for o in range(nblanks)]

    keywords = [set([(None, RANDOM_TAG)]) if o != "V" else set() for o in args_to_blank]

    # span range
    start_range = start if start is not None and start > 0 else 0
    end_range = end if end is not None and end <= len(doc) else len(doc)
    curr_idxes, curr_tag = [-1, -1], ""
    tag_word_tuple = list(zip(raw_tags, words))  # [start:end]
    end = len(tag_word_tuple)
    for idx_, (tag, word) in enumerate(tag_word_tuple):
        # already saved a tag and now it ends
        # idx = start + idx_
        idx = idx_
        if curr_idxes[0] != -1 and (not tag.startswith("I-") or idx_ == len(tag_word_tuple) - 1):
            # if it's the last token and it's still going, then add one
            if idx_ == len(tag_word_tuple) - 1 and tag.startswith("I-"):
                curr_idxes[1] += 1
            # where in the args
            order = args_to_blank.index(curr_tag)
            # if already encountered tag, keep first; TODO: be able to encode multiple arguments?
            if tags[order] != EMPTY_FILL:
                continue
            # exclude referents
            if not curr_tag.startswith("R-") and (
                curr_tag == "V" or (curr_idxes[0] >= start_range and curr_idxes[1] <= end_range)
            ):
                # update the info
                curr_span = doc[curr_idxes[0] : curr_idxes[1]]
                blank_indexes[order] = deepcopy(curr_idxes)
                tags[order] = curr_tag
                if curr_tag == "V":
                    if vlemma is None:
                        vlemma = curr_span.lemma_
                    keyword = vlemma
                else:
                    keyword = get_keyword_for_span(curr_span, keyword_str)
                keywords[order].add(keyword)
            # reset
            curr_idxes, curr_tag = [-1, -1], ""
            # if only save the first
            if is_extract_first:
                break
        # if is the starting point of an argument to be blanked
        if any([tag.startswith(f"B-{t}") for t in args_to_blank]):
            curr_idxes = deepcopy([idx, idx + 1])
            curr_tag = clean_prefix_for_one_tag(tag)
        elif tag.startswith(f"I-{curr_tag}"):
            curr_idxes[1] = idx + 1
    # happens when last token ends in B-; need to add separately
    # (bc args updated after first occurrence in above code)
    if curr_tag != "":
        # assert raw_tags[-1].startswith("B-")
        if raw_tags[-1].startswith("B-"):
            order = args_to_blank.index(curr_tag)
            # exclude referents
            # if already encountered tag, keep first; TODO: be able to encode multiple arguments?
            # TODO: changed tags[order] != EMPTY_FILL to tags[order] == EMPTY_FILL
            if (
                not curr_tag.startswith("R-")
                and tags[order] == EMPTY_FILL
                and (
                    curr_tag == "V" or (curr_idxes[0] >= start_range and curr_idxes[1] <= end_range)
                )
            ):
                # update the info
                curr_span = doc[curr_idxes[0] : curr_idxes[1]]
                blank_indexes[order] = deepcopy(curr_idxes)
                tags[order] = curr_tag
                if curr_tag == "V":
                    if vlemma is None:
                        vlemma = curr_span.lemma_
                    keyword = vlemma
                else:
                    keyword = get_keyword_for_span(curr_span, keyword_str)
                keywords[order].add(keyword)
    # reset
    answers = [EMPTY_FILL for o in range(nblanks)]
    short_tags = [EMPTY_FILL for o in range(nblanks)]
    assert len(answers) == len(tags)

    for order, (blank_idx, tag) in enumerate(zip(blank_indexes, tags)):
        bl_start, bl_end = blank_idx
        # does not exist
        if bl_start == -1 or bl_end == -1:
            continue
        curr_span = doc[bl_start:bl_end]
        answers[order] = curr_span.text
        control_code = (
            convert_tag2readable(vlemma, tag, frameset_id, frameset_path=frameset_path)
            if tag != "V"
            else "VERB"
        )
        if answers[order] != EMPTY_FILL:
            answers[order] = f"[{control_code}: {answers[order]}]"
        short_tags[order] = control_code
    return keywords, blank_indexes, answers, tags, short_tags


def overblank(doc, blank_indexes, tags):
    # start with examples that are not core
    cores = ["ARG0", "ARG1", "V"]
    sorted_tags = sorted(tags, key=lambda t: cores.index(t) if t in cores else -1)
    overblank_indexes_lst = []
    for tag in sorted_tags:
        idx = tags.index(tag)
        bl_start, bl_end = blank_indexes[idx]
        # blank does not exist yet
        if bl_start == -1 or bl_end == -1:
            continue
        curr_span = doc[bl_start:bl_end]
        # extend based on some probability
        if tag == "V":
            pextend = 0.2
        elif tag in ["ARG0", "ARG1"]:
            pextend = 0.2
        else:
            pextend = 0.5
        is_extend = np.random.choice([True, False], p=[pextend, 1 - pextend])
        # if not extend just go
        if not is_extend:
            continue
        # this works bc blank_indexes and tags should be in same order?
        excluded = list(np.concatenate(list([range(b[0], b[1]) for b in blank_indexes])))
        root = curr_span.root.head

        left_edge, right_edge = root.left_edge.i, root.right_edge.i + 1
        # get min/max surrounding the current span
        le_range = [idx for idx in excluded if idx < bl_start]
        re_range = [idx for idx in excluded if idx > bl_end - 1]
        min_left_edge = max(le_range) + 1 if le_range else left_edge
        max_right_edge = min(re_range) if re_range else right_edge

        left_edge = max(left_edge, min_left_edge)
        right_edge = min(right_edge, max_right_edge)

        overblank_indexes = [
            idx
            for idx in range(int(left_edge), int(right_edge))
            if idx not in excluded and idx not in overblank_indexes_lst
        ]
        overblank_indexes_lst.extend(overblank_indexes)

    consecutive_indexes = [list(group) for group in mit.consecutive_groups(overblank_indexes_lst)]
    blank_indexes = [[min(span), max(span) + 1] for span in consecutive_indexes]
    answers = [" ".join([doc[idx].text for idx in group]) for group in consecutive_indexes]

    return blank_indexes, answers


def get_possible_blank_idxes(verb):
    """Get possible places to insert blank. Must be between sub- parsing trees.
    Args:
        verb (Token): the verb token
    Returns:
        List[int]: all possible indexes for inserting blanks
    """
    indexes = set()
    # doc = verb.doc
    indexes.add(verb.i)
    for t in verb.children:
        if t.is_punct:
            continue
        indexes.add(t.left_edge.i)
        indexes.add(t.right_edge.i + 1)
    return list(indexes)


def get_excluded_blank_indices(blank_indexes, possible_blank_indexes):
    """Helper function for add_extra_blanks.
    Called if no_empty_blanks_at_start = True.
    Returns indices that cannot have empty blank indexes.
    """
    end_blank_indexes = [b[1] for b in blank_indexes]
    excluded = [0]
    for idx in possible_blank_indexes:
        if idx == 0:
            continue
        continuing_start_blanks = all(
            temp_idx in end_blank_indexes for temp_idx in range(1, idx + 1)
        )
        if continuing_start_blanks:
            excluded.append(idx)
    return excluded


def add_extra_blanks(
    verb,
    blank_indexes,
    no_empty_blanks_at_start=False,
    extra_blank_start=None,
    extra_blank_end=None,
):
    """After extracting existing places to blank, if there's still unselected
    blank places ([-1, -1] from `extract_exist_blanks`), randomly select some.
    Args:
        verb (Token): the spacy verb token
        blank_indexes ([int, int][]): the blanked indexes.
            If an entry is unfilled, it'd be [-1, -1]
    Returns:
        [int, int][]: filled indexes with no more [-1, -1]
    """
    vidx = verb.i
    nblanks = len(blank_indexes)
    possible_blank_indexes = get_possible_blank_idxes(verb)
    # get possible places to insert additional blanks -- between two subtrees
    # exclude blanks that'd overlap with existing ones.
    excluded = list(np.concatenate(list([range(b[0] + 1, b[1]) for b in blank_indexes])))

    # if no empty blanks at start, we exclude index 0 AND
    if no_empty_blanks_at_start:
        # excluded.append(0)
        indexes_at_start = get_excluded_blank_indices(blank_indexes, possible_blank_indexes)
        excluded.extend(indexes_at_start)
    if extra_blank_start is not None:
        excluded += list(range(extra_blank_start))
    if extra_blank_end is not None:
        excluded += list(range(extra_blank_end, len(verb.doc)))

    insert_indexes = [i for i in possible_blank_indexes if i not in excluded]
    # go through all the args, if one does not exist, randomly add indexes
    exist_insert_idxes = [i[0] for idx, i in enumerate(blank_indexes) if i[0] != -1]
    # if not all argument is included
    if len(exist_insert_idxes) < nblanks:
        select_count = nblanks - len(exist_insert_idxes)
        if len(insert_indexes) == 0:
            # no more candidates
            return blank_indexes
        attempt, max_attempt = 0, 100
        selected = []
        # try to make the blank occur in both directions of the verb
        while (
            not selected
            or (
                all([i <= vidx for i in exist_insert_idxes + selected])
                or all([i >= vidx for i in exist_insert_idxes + selected])
            )
        ) and attempt < max_attempt:
            selected = list(
                np.random.choice(
                    insert_indexes, select_count, replace=select_count > len(insert_indexes)
                )
            )
            attempt += 1
        curr_select_idx = 0
        # insert the selected indexes for blankingf
        for idx, b in enumerate(blank_indexes):
            if b[0] == -1:
                blank_indexes[idx] = [selected[curr_select_idx], selected[curr_select_idx]]
                curr_select_idx += 1
    return blank_indexes


def gen_prompt_and_answer_strings(
    doc, blank_indexes, blank_appearance_indexes, answers, return_sequence=True
):
    """Translate the extracted answers and blanks into prompt and answer
    string. This is a helper function for the prompt generation funcs.
    Args:
        doc (Doc): original doc sentence
        blank_indexes ([int, int][]): blanks
        answers (str[]): List of groundtruth fillins corresponding to blank_indexes
        return_sequence (bool): Whether to return the whole output sequence as the target
        (vs. blanking structure)
    Returns:
        [str, str]: the prompt with blanks, and the corresponding answer
    """
    # create the prompt
    indxes, appearances = [], []
    # get the order of the blanks to get <extra_id_[0-n]>
    blank_indexes_order = list(range(len(blank_indexes)))
    blank_indexes_order = sorted(
        blank_indexes_order, key=lambda i: (blank_indexes[i][0], blank_indexes[i][1])
    )
    for id_idx, order in enumerate(blank_indexes_order):
        indxes.append(blank_indexes[order])
        appearances.append(blank_appearance_indexes[order])
    prompt = flatten_fillins(doc, indxes, appearances, is_return_text=True)
    # reorder the answering
    if answers is not None and len(answers) == len(blank_indexes):
        answers = [answers[order] for order in blank_indexes_order]
        if return_sequence:
            answer = prompt
            for i, t in enumerate(answers):
                answer = answer.replace(f"{convert_idx2blank(i)}", t)
        else:
            answer = (
                " ".join([f"{convert_idx2blank(i)} {t}" for i, t in enumerate(answers)])
                + " "
                + convert_idx2blank(id_idx + 1)
            )
    else:
        answer = ""

    # TODO: hacky, but clean up spaces
    while "  " in answer:
        answer = answer.replace("  ", " ").strip()
    return prompt, answer


def gen_random_prompt(
    doc,
    frameset_id,
    raw_tags,
    vlemma=None,
    return_prompt_type="sparse",
    nblanks=None,
    return_sequence=True,
    frameset_path=DEFAULT_FRAME_SET_PATH,
    # keyword_str="NOUN_CHUNKS,RANDOM_SUBTREES,EXACT,PREFIX",
    keyword_str="EXACT,UNCASED",
    p_overblank=0.0,
):
    """Get prompt with randomly chosen subset of tags
        Blanks subset of tags 20% of the time; all tags 80% of the time
        Always blanks verb
    Used for generating prompts at train time
    Calls gen_prompts_by_tags with either subset of tags or all tags.
    """

    is_subset = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
    args_to_blank = get_unique_tags(raw_tags)
    if is_subset:
        blank_proportion = random()
        num_blank_args = int(blank_proportion * len(args_to_blank))
        args_to_blank = np.random.choice(args_to_blank, num_blank_args, replace=False)
    else:
        num_blank_args = len(args_to_blank)
    # always blank verb;
    # TODO: verb included in header even if not blanked--do we want this behavior?
    if "V" not in args_to_blank:
        args_to_blank = np.append(args_to_blank, "V")
    prompts = gen_prompts_by_tags(
        doc,
        frameset_id,
        raw_tags,
        args_to_blank=args_to_blank,
        return_prompt_type=return_prompt_type,
        return_sequence=return_sequence,
        frameset_path=frameset_path,
        p_overblank=p_overblank,
        vlemma=vlemma,
        nblanks=nblanks,
    )
    # print(prompts)
    if prompts:
        return_prompt_idx = np.random.choice(len(prompts), 1)[0]
        return_prompt = prompts[return_prompt_idx]  # np.random.choice(prompts, 1)[0]
        return_prompt.is_subset = is_subset
        return_prompt.num_blank_args = num_blank_args
        return return_prompt
    return None


def get_unique_tags(raw_tags):
    """Helper function to get possible tags to blank from raw_tags
        Useful for generating prompts at train time.
    Args:
        raw_tags (List[str]): the array of SRL tags.
    Returns:
        List[str]: list of unique raw tags with cleaned prefixes
            (i.e. B-ARGM-TMP -> ARGM-TMP)
    """
    target_tags = set([clean_prefix_for_one_tag(t) for t in raw_tags])
    target_tags = [t for t in target_tags if t not in ["O"]]
    return target_tags


# TODO: add capabilities to add empty blanks to start of sentences


def gen_prompts_by_tags(
    doc,
    frameset_id,
    raw_tags,
    # keyword_str="NOUN_CHUNKS,RANDOM_SUBTREES,EXACT,PREFIX",
    keyword_str="EXACT,UNCASED",
    args_to_blank=None,
    short_args_to_blank=None,
    nblanks=None,
    vlemma=None,
    return_prompt_type="sparse",
    return_sequence=True,
    is_blank_aux=True,
    no_empty_blanks_at_start=False,
    frameset_path=DEFAULT_FRAME_SET_PATH,
    p_overblank=0.0,
    start=None,
    end=None,
    extra_blank_start=None,
    extra_blank_end=None,
):
    """Get prompts for the modifiers.
    Args:
        doc (Doc): spacy doc
        frameset_id (str): frameset id of verb.
            Used for getting appropriate raw tag -> short tag mapping
        raw_tags (List[str]): the array of the SRL tags. It should have one and only one B-V
            denoting the targeted predicate
        keyword_str (str): string specifying types of keywords to include
            See parse_keyword_type and get_keyword_candidates_for_span for more information
        args_to_blank Optional(List[str]): the tags to extract for the header ctrl, in the SRL format
            (not already readable)
            If not supplied, blank all args
        short_args_to_blank: readable tags to extract for header
            Provides further filter on args_to_blank for which args to keep
        return_prompt_type (str): "concrete"|"all"|"sparse" whether to return those with the most
            random tag, the least, or all possible
        frameset_path (str, optional): the path to frameset. See `convert_tag2readable`
        no_empty_blanks_at_start (bool, optional): Prevent adding empty blanks at start
            Defaults to False, i.e. we can put empty blanks at start
        nblanks (int): the number of blanks to put.
            Defaults to the max of: randomly chosen integer between 1, MAX_NUM_BLANKS and
                the number of args_to_blank.
            If supplied, will override the random choosing such that num blanks is max of
                supplied nblanks and number of args_to_blank.
        p_overblank: probability of computing overblank.
    Returns:
        If type is "all", will be a list of the dict; otherwise, one dict in
        the following format:
        Dict[str, str]: {
            prompt: [VERB+active: serve | AGENT: Jews | PATIENT: * | TEMPORAL: day]
                    <extra_id_0> For this hope <extra_id_1> <extra_id_2> <extra_id_3>
                    <extra_id_4> God <extra_id_5> <extra_id_6> .
            answer: <extra_id_0>  <extra_id_1> [AGENT: the Jews] <extra_id_2>  <extra_id_3>
                    [VERB: serve] <extra_id_4>  <extra_id_5> [TEMPORAL: day and night]
                    <extra_id_6>  <extra_id_7>
            meta: Munch object with keys noncore_args, blank_indexes, answers, patient,
                  agent, frameset_id, vvoice, vlemma, doc
                example:
                Munch(
                    {
                        'noncore_args': [
                            Munch({'tlemma': 'day', 'raw_tag': 'ARGM-TMP', 'tag': 'TEMPORAL',
                            'blank_idx': [7, 10]})],
                        'blank_indexes': [[3, 5], [6, 6], [5, 6], [7, 10], [7, 7], [10, 10], [3, 3]],
                        'agent': 'Jews', 'patient': '*',
                        'answers': ['[AGENT: the Jews]', '', '[VERB: serve]',
                            '[TEMPORAL: day and night]', '', '', ''],
                        'vvoice': 'active', 'vlemma': 'serve',
                        'doc': For this hope the Jews serve God day and night . ,
                        'raw_tags': ['B-ARGM-PRP', 'I-ARGM-PRP', 'I-ARGM-PRP', 'B-ARG0',
                            'I-ARG0', 'B-V', 'B-ARG2', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP', 'O'],
                        'frameset_id': '01'})
        }
    """
    vidx = get_vindex_by_tags(raw_tags)
    verb = doc[vidx]
    verb_voice = get_verb_voice(verb)
    verb_tense = get_verb_tense(verb, doc)
    if vlemma is None:
        vlemma = doc[vidx].lemma_
    if vidx is None:
        warnings.warn("Returning None because vidx is None")
        return None
    verb = doc[vidx]
    if args_to_blank is not None:
        if "V" not in args_to_blank:
            raise ValueError("'V' must be in args_to_blank; must always blank verb")
    if short_args_to_blank is not None:
        if "VERB" not in short_args_to_blank:
            raise ValueError("'VERB' must be in short_args_to_blank; must always blank verb")
    if args_to_blank is None:
        args_to_blank = get_unique_tags(raw_tags)
    short_tags = [
        convert_tag2readable(vlemma, target_tag, frameset_id, frameset_path=frameset_path)
        for target_tag in args_to_blank
    ]
    # keep only those for which we find the semantically meaningful name
    args_to_blank = [arg for arg, st in zip(args_to_blank, short_tags) if st is not None]
    short_tags = [st for st in short_tags if st is not None]
    if short_args_to_blank is not None:
        args_to_blank = [
            arg for arg, st in zip(args_to_blank, short_tags) if st in short_args_to_blank
        ]
        short_tags = [st for st in short_tags if st in short_args_to_blank]
    if not short_tags:
        warnings.warn("Returning None because short_tags is None")
        return None
    # if nblanks not supplied, randomly choose between 1-MAX_NUM_BLANKS # to blank
    if nblanks is None:
        nblanks = np.random.choice(MAX_NUM_BLANKS, 1)[0]
    # make sure enough blanks for args to blank
    nblanks = max(nblanks, len(args_to_blank))
    _, temp_raw_tags = [d.text for d in doc], deepcopy(raw_tags)

    keywords, blank_indexes, answers, ordered_tags, ordered_short_tags = extract_exist_blanks(
        doc,
        temp_raw_tags,
        args_to_blank,
        frameset_id,
        nblanks=nblanks,
        is_extract_first=False,
        keyword_str=keyword_str,
        start=start,
        end=end,
        vlemma=vlemma,
        frameset_path=frameset_path,
        p_overblank=p_overblank,
    )
    if is_blank_aux:
        aux_blank_indexes, aux_answers = extract_aux_blanks(verb)
        blank_indexes.extend(aux_blank_indexes)
        answers.extend(aux_answers)

    is_overblank = np.random.choice([True, False], p=[p_overblank, 1 - p_overblank])
    if is_overblank:
        overblank_indexes, overblank_answers = overblank(doc, blank_indexes, ordered_tags)
        blank_indexes.extend(overblank_indexes)
        answers.extend(overblank_answers)

    blank_indexes = add_extra_blanks(
        verb,
        blank_indexes,
        no_empty_blanks_at_start=no_empty_blanks_at_start,
        extra_blank_start=extra_blank_start,
        extra_blank_end=extra_blank_end,
    )
    blank_appearance_indexes = [start for start, _ in blank_indexes]
    prompt, answer = gen_prompt_and_answer_strings(
        doc, blank_indexes, blank_appearance_indexes, answers, return_sequence=return_sequence
    )

    metas = []
    # unique_tlemmas is list of tuples representing all possible combinations of keywords,
    # same order as tags
    agent_idx = ordered_short_tags.index("AGENT") if "AGENT" in ordered_short_tags else None
    patient_idx = ordered_short_tags.index("PATIENT") if "PATIENT" in ordered_short_tags else None
    agent_raw_tag = ordered_tags[agent_idx] if agent_idx is not None else None
    patient_raw_tag = ordered_tags[patient_idx] if patient_idx is not None else None

    unique_tlemmas = set(itertools.product(*keywords))
    for unique_tlemmas in itertools.product(*keywords):
        noncore_args = [
            Munch(
                tlemma=unique_tlemmas[idx][1],
                tlemma_type=unique_tlemmas[idx][0],
                raw_tag=raw_tag,
                tag=short_tag,
                blank_idx=blank_idx,
            )
            for idx, (raw_tag, short_tag, blank_idx) in enumerate(
                zip(ordered_tags, ordered_short_tags, blank_indexes)
            )
            if raw_tag not in [agent_raw_tag, patient_raw_tag, "V", ""]
        ]
        core_args = []
        for (raw_tag, arg_short_tag, eidx) in [
            (agent_raw_tag, "AGENT", agent_idx),
            (patient_raw_tag, "PATIENT", patient_idx),
        ]:
            if eidx is None:
                continue
            blank_idxes = [
                idx
                for idx, short_tag in zip(blank_indexes, ordered_short_tags)
                if short_tag == arg_short_tag
            ]
            blank_idx = blank_idxes[0] if blank_idxes else None
            core_args.append(
                Munch(
                    tlemma=unique_tlemmas[eidx][1],
                    tlemma_type=unique_tlemmas[eidx][0],
                    raw_tag=raw_tag,
                    tag=arg_short_tag,
                    blank_idx=blank_idx,
                )
            )
        # set as None when agent or patient does not exist in prompt
        # TODO: is this behavior we want? (before used RANDOM_TAG)
        metas.append(
            Munch(
                noncore_args=noncore_args,
                # solve so agent always appear first
                core_args=sorted(core_args, key=lambda c: c.raw_tag),
                blank_indexes=blank_indexes,
                blank_appearance_indexes=blank_appearance_indexes,
                answers=answers,
                vvoice=verb_voice,
                vlemma=vlemma,
                vtense=verb_tense,
                doc=doc,
                raw_tags=raw_tags,
                frameset_id=frameset_id,
            )
        )
    # get the header
    results = [
        Munch(prompt=f"{gen_header_by_meta(meta)} {prompt}", answer=answer, meta=meta)
        for meta in metas
    ]
    results = sorted(
        results,
        key=lambda r: sum(
            [int(arg.tlemma == RANDOM_TAG) for arg in r.meta.noncore_args + r.meta.core_args]
        ),
    )
    if results == []:
        warnings.warn(f"No prompts created; unique_tlemmas: {unique_tlemmas}")
        return None
    if return_prompt_type == "all":
        return results
    else:
        # first one is the most concrete one, last one is the most sparse
        return results[0] if return_prompt_type == "concrete" else results[-1]


######################################################################
# prompt generation utils
######################################################################

ROOT = "ROOT"
OPENIE = "OPENIE"
CONNECT = "CONNECT"
NOUN_CHUNKS = "NOUN_CHUNKS"
RANDOM_SUBTREES = "RANDOM_SUBTREES"
EXACT = "EXACT"
PREFIX = "PREFIX"
UNCASED = "UNCASED"


def uppercase_keyword(keyword):
    """Helper function to uppercase keyword str"""
    try:
        keyword = keyword[0].upper() + keyword[1:]
    except IndexError:
        warnings.warn(f"IndexError in uppercasing '{keyword}'. Returning original kw.")
    return keyword


def lowercase_keyword(keyword):
    """Helper function to lowercase keyword str if it starts with stopword"""
    try:
        split_keyword = keyword.split(" ")
        if split_keyword[0].lower() in stopwords.words("english"):
            keyword = " ".join([split_keyword[0].lower()] + split_keyword[1:])
    except IndexError:
        warnings.warn(f"IndexError in lowercasing '{keyword}'. Returning original kw.")
    return keyword


# TODO: make more robust
def fix_pronoun_form(keyword, target="subj"):
    """Helper function to change change form of pronouns to obj/subj form
    Called by passivize/unpassivize_keyword() functions
    For ex, if we change agent 'they' to agent in passive construction 'by them',
    need to change 'they' to 'them'"""
    if target not in ["obj", "subj"]:
        raise ValueError

    subj_to_obj = {"they": "them", "she": "her", "he": "him", "we": "us", "i": "me"}

    obj_to_subj = {v: k for k, v in subj_to_obj.items()}

    words = keyword.split(" ")
    new_words = [None] * len(words)
    for idx, word in enumerate(words):
        word_is_lower = word[0].islower()
        word_key = word.lower()
        if target == "obj" and word_key in subj_to_obj:
            new_word = subj_to_obj[word_key]
        elif target == "subj" and word_key in obj_to_subj:
            new_word = obj_to_subj[word_key]
        else:
            new_word = word
        new_words[idx] = new_word if word_is_lower else uppercase_keyword(new_word)
    new_keyword = " ".join(new_words)
    return new_keyword


def unpassivize_keyword(keyword):
    """Helper function to change keyword agent for passive to *not*,
    i.e. remove "by" or "By" from start of keyword"""
    if keyword.startswith("by ") or keyword.startswith("By "):
        is_upper = keyword.startswith("By")
        keyword = fix_pronoun_form(keyword, target="subj")
        keyword = keyword[3:]
        if is_upper:
            keyword = keyword[0].upper() + keyword[1:]
    return keyword


def passivize_keyword(keyword):
    """Helper function to change keyword agent for passive to *not*,
    i.e. remove "by" or "By" from start of keyword"""
    # TODO: this will improperly uppercase if keyword is named entity--fix
    try:
        if keyword and (not keyword.startswith("by ") and not keyword.startswith("By ")):
            prepend_tok = "By " if keyword[0].isupper() else "by "
            keyword = fix_pronoun_form(keyword, target="obj")
            keyword = prepend_tok + keyword.lower()
    except IndexError:
        warnings.warn(f"IndexError in passivizing '{keyword}'. Returning original kw.")
    return keyword


def is_lowercase(keyword):
    """Helper function to determine whether keyword starts with lowercase"""
    try:
        return keyword[0].islower()
    except IndexError:
        warnings.warn(f"IndexError in calling is_lowercase on '{keyword}'. Returning False.")
    return False


def parse_keyword_type(keyword_str):
    """A parser that parses keyword type for get_keyword_candidates_for_span()"""

    is_include_root = ROOT in keyword_str
    is_include_openie = OPENIE in keyword_str
    is_include_connect = CONNECT in keyword_str
    is_include_noun_chunks = NOUN_CHUNKS in keyword_str
    is_include_random_subtrees = RANDOM_SUBTREES in keyword_str
    is_include_exact = EXACT in keyword_str
    is_include_prefix = PREFIX in keyword_str
    is_uncased = UNCASED in keyword_str

    keyword_type = Munch(
        is_include_root=is_include_root,
        is_include_openie=is_include_openie,
        is_include_connect=is_include_connect,
        is_include_noun_chunks=is_include_noun_chunks,
        is_include_random_subtrees=is_include_random_subtrees,
        is_include_exact=is_include_exact,
        is_include_prefix=is_include_prefix,
        is_uncased=is_uncased,
    )
    assert any(
        [
            is_include_root,
            is_include_openie,
            is_include_connect,
            is_include_noun_chunks,
            is_include_random_subtrees,
            is_include_exact,
            is_include_prefix,
        ]
    ), (
        f"Could not parse given keyword string ({keyword_str})"
        + "\n"
        + "Must include at least one of the following: "
        + f"{[ROOT, OPENIE, CONNECT, NOUN_CHUNKS, RANDOM_SUBTREES, EXACT, PREFIX]}"
    )

    return keyword_type


def gen_prompt_by_replace_header(orig_prompt, target_meta, frameset_path=DEFAULT_FRAME_SET_PATH):
    """Create prompts through header replacement.
    This is for creating prompts during generation time.
    Args:
        orig_prompt (str): original prompt
        target_meta (str): target prompt
            Will replace original header with header created using this meta
    Returns:
        Munch[]: {
            prompt: str, the generated prompt
            meta: the corresponding meta
        }[]
    """
    new_header = gen_header_by_meta(target_meta)
    new_prompt = re.sub("\[[^\]]+\]", new_header, orig_prompt, 1)  # noqa:W605

    # delete, since this meta info is not true for new prompt
    del target_meta.blank_indexes
    del target_meta.answers
    return Munch(
        prompt=new_prompt,
        meta=target_meta,
    )


def gen_prompt_and_answer_by_meta(meta, return_sequence=True):
    """Helper function for gen_prompt_by_perturb_meta
    Creates a prompt based on meta information

    """
    header = gen_header_by_meta(meta)
    prompt, answer = gen_prompt_and_answer_strings(
        meta.doc,
        meta.blank_indexes,
        meta.blank_appearance_indexes,
        meta.answers,
        return_sequence=return_sequence,
    )
    return f"{header} {prompt}", answer


def shuffle_empty_blank_indexes(meta):
    # TODO: add capability to exclude shuffling such that empty blanks are at start
    """Helper function that randomly shuffles blank_indexes in meta
    Used for gen_prompt_by_perturb_meta for changing indexes
    Args:
        meta
    Returns:
        meta: new meta with shuffled blank indexes
    """

    vidx = get_vindex_by_tags(meta.raw_tags)
    verb = meta.doc[vidx]
    # if start == end, this is empty blank index
    empty_blanks = [
        idx
        for idx, (bl_start, bl_end) in enumerate(meta.blank_appearance_indexes)
        if bl_start == bl_end
    ]
    possible_idxes = get_possible_blank_idxes(verb)
    # TODO: allow replacement--is this robust?
    new_empty_idxes = np.random.choice(possible_idxes, len(empty_blanks), replace=True)
    # update empty blank indexes with new randomly chosen locations
    empty_counter = 0
    new_blank_idxes = []
    for idx, orig_span in enumerate(meta.blank_appearance_indexes):
        if idx in empty_blanks:
            blank_span = [new_empty_idxes[empty_counter], new_empty_idxes[empty_counter]]
            empty_counter += 1
        else:
            blank_span = orig_span
        new_blank_idxes.append(blank_span)
    new_meta = meta.copy()
    new_meta.blank_appearance_indexes = new_blank_idxes
    new_meta.blank_indexes = new_blank_idxes
    return new_meta


def sample_common_keyword(key_dicts, label, keyword_type, orig_keyword, top_k=15):
    """Helper function for gen_prompt_by_perturb_meta() when target keyword/content not given
        Used for UL training for keyword following
        Randomly samples common keyword for a given tag
    Args:
        key_dicts (dict(str, dict(str, Counter))): dict of sub-dicts
            See output of get_common_keywords_by_tag()
        label (str): label for which to sample keyword
            In control code format (vs. SRL raw tag)
        keyword_type (str): one of complete/partial/sparse
        orig_keyword (str): original lemma
            Needed to make sure we do not re-sample
            Make sure sampled keyword (returned) is not subset of original keyword
        top_k (int, optional): used to select keywords to sample from
            i.e., randomly sample keyword from top_k most frequent keywords in key_dicts[label]
    Returns:
        str: keyword
    """
    keyword = RANDOM_TAG
    try:
        if label not in key_dicts or keyword_type not in key_dicts[label]:
            return RANDOM_TAG  # TODO: change default behavior here
        keyword_candidates = Counter(key_dicts[label][keyword_type]).most_common(top_k)
        num_tries = 0
        successful_sample = False
        while not successful_sample and num_tries < 5 and len(keyword_candidates):
            keyword_idx = np.random.choice(len(keyword_candidates), 1)[0]
            keyword = keyword_candidates[keyword_idx][0]
            successful_sample = keyword not in orig_keyword
            num_tries += 1
    except ValueError as e:
        error_message = (
            str(e)
            + "\n"
            + f"Error arose for label ({label}),"
            + f"keyword_type ({keyword_type}) in key_dicts[{label}][{keyword_type}]"
            + f"{key_dicts[label][keyword_type]}"
        )
        print(error_message)
        return RANDOM_TAG  # TODO: change default behavior here
    return keyword


def gen_prompt_by_perturb_str(
    doc, tags, perturb_str, base_meta, frameset_path=DEFAULT_FRAME_SET_PATH
):
    """Generates perturbed prompt based on perturb_str
    Calls parse_change_type_meta and gen_prompt_by_perturb_meta
    Helpful at generation time
    Args:
        doc (Doc): the spacy processed doc
        tags (str[]): per-token tags in BIO format.
        perturb_str (Munch): the str for perturbation.
            See input of `parse_change_type_meta`
        base_meta ([Munch]):
            {prompt: str, meta: meta}
    Returns:
        Munch[]: {
            prompt: str, the generated prompt
            meta: the corresponding meta
    """

    perturb_meta = parse_change_type_meta(perturb_str)
    perturbed = gen_prompt_by_perturb_meta(
        doc, tags, perturb_meta, base_meta, frameset_path=frameset_path
    )
    return perturbed


def get_core_idxes_from_meta(meta):
    agent_eidxes = [idx for idx, c in enumerate(meta.core_args) if c.tag == "AGENT"]
    patient_eidxes = [idx for idx, c in enumerate(meta.core_args) if c.tag == "PATIENT"]
    agent_eidx = agent_eidxes[0] if agent_eidxes else None
    patient_eidx = patient_eidxes[0] if patient_eidxes else None
    return Munch(aidx=agent_eidx, pidx=patient_eidx)


def gen_prompt_by_perturb_meta(
    doc,
    raw_tags,
    perturb_meta,
    base_meta,
    return_sequence=True,
    is_training=False,
    common_keywords_by_tag=None,
    frameset_path=DEFAULT_FRAME_SET_PATH,
):

    """Create prompts based on perturb_meta contrl.
    This is for creating prompts during generation time.
    Args:
        doc (Doc): the spacy processed doc
        raw_tags (str[]): per-token tags in BIO format.
        perturb_meta (Munch): the meta info for perturbation.
            See output of `parse_change_type_meta`
        base_meta ([Munch]):
            {prompt: str, meta: meta}
        return_sequence (bool):
            whether to return target ("answers") as sequence
        is_training (bool): whether or not perturbed prompts are being used during training
            used to decide whether to insert "MODAL" for perturbing tense to future
        common_keywords_by_tag (dict, optional):
            dict of sub-dicts specifying keywords by tag/keyword_type
            only needed for changing content, i.e.:

    Returns:
        Munch[]: {
            prompt: str, the generated prompt
            meta: the corresponding meta
            perturb_meta: the input perturb_meta
        }[]
        common_keywords_by_tag (dict, optional):
            dict of sub-dicts specifying keywords by tag/keyword_type
            only needed for changing content, i.e.:

    Returns:
        Munch[]: {
            prompt: str, the generated prompt
            meta: the corresponding meta
            perturb_meta: the input perturb_meta
        }[]
    """
    if not perturb_meta:
        return None
    vidx = get_vindex_by_tags(raw_tags)
    if vidx is None:
        return None
    vlemma = base_meta.vlemma
    # generated_prompts = {}
    # TODO: get rid of the is_core arg (can change vform for both core and noncore)
    # new_metas = []

    context_changes = perturb_meta.context_changes
    core_changes = perturb_meta.core_changes
    noncore_changes = perturb_meta.noncore_changes
    verb_changes = perturb_meta.verb_changes

    # need to have a base
    base_prompt, _ = gen_prompt_and_answer_by_meta(base_meta, return_sequence=return_sequence)
    new_meta = base_meta.copy()

    if verb_changes.is_change_vvoice:
        voice_mapper = {"passive": "active", "active": "passive"}
        new_voice = (
            verb_changes.target_voice
            if verb_changes.target_voice
            else voice_mapper[base_meta.vvoice]
        )
        new_meta.vvoice = new_voice

    if verb_changes.is_change_vtense:
        tense_options = [
            tense for tense in ["present", "past", "future"] if tense != base_meta.vtense
        ]
        new_tense = (
            verb_changes.target_tense
            if verb_changes.target_tense
            else np.random.choice(tense_options, 1)[0]
        )
        new_meta.vtense = new_tense
        # TODO: fix hack: if changing to future, need to make sure MODAL tag exists, since "future"
        # only returned by get_verb_tense if "will" or "shall" (modals) are in prompt header
        # only do this during generation time
        if new_tense == "future" and not is_training:
            modal_idx = [idx for idx, arg in enumerate(new_meta.noncore_args) if arg.tag == "MODAL"]
            # if modal already in header, change keyword to * to generate "will" or "shall"
            if modal_idx:
                modal_idx = modal_idx[0]
                new_meta.noncore_args[modal_idx].tlemma = RANDOM_TAG
                new_meta.noncore_args[modal_idx].tlemma_type = None
            else:
                # TODO: blank_idx=None because won't be used, but make sure this is robust
                new_meta.noncore_args.append(
                    Munch(
                        {
                            "tlemma": RANDOM_TAG,
                            "tlemma_type": None,
                            "raw_tag": "ARGM-MOD",
                            "tag": "MODAL",
                            "blank_idx": None,
                        }
                    )
                )

    if verb_changes.is_change_vlemma:
        new_vlemma = (
            verb_changes.target_vlemma
            if verb_changes.target_vlemma is not None
            else sample_common_keyword(common_keywords_by_tag, "VERB", "complete", new_meta.vlemma)
        )
        new_meta.vlemma = new_vlemma

    assert not (
        core_changes.is_swap_core
        and (
            core_changes.agent_changes.is_change_content_type
            or core_changes.patient_changes.is_change_content_type
        )
    ), "Cannot change agent/patient content if swapping core"

    core_idx = get_core_idxes_from_meta(new_meta)
    if core_changes.is_swap_core and core_idx.aidx is not None and core_idx.pidx is not None:
        # identify the idxes
        # assert core_idx.aidx is not None and core_idx.pidx is not None
        agent_raw_tag = new_meta.core_args[core_idx.aidx].raw_tag
        patient_raw_tag = new_meta.core_args[core_idx.pidx].raw_tag
        new_meta.core_args[core_idx.aidx].tag, new_meta.core_args[core_idx.aidx].raw_tag = (
            "PATIENT",
            patient_raw_tag,
        )
        new_meta.core_args[core_idx.pidx].tag, new_meta.core_args[core_idx.pidx].raw_tag = (
            "AGENT",
            agent_raw_tag,
        )
        new_meta.core_args = sorted(new_meta.core_args, key=lambda c: 0 if c.tag == "AGENT" else 1)
        # TODO: is this robust? better to actually encode agent/patient indices when creating answers
        agent_idx = [idx for idx, ans in enumerate(new_meta.answers) if "AGENT" in ans]
        patient_idx = [idx for idx, ans in enumerate(new_meta.answers) if "PATIENT" in ans]
        temp_answers = deepcopy(new_meta.answers)
        if agent_idx:
            agent_idx = agent_idx[0]
            temp_answers[agent_idx] = temp_answers[agent_idx].replace("AGENT", "PATIENT")
        if patient_idx:
            patient_idx = patient_idx[0]
            temp_answers[patient_idx] = temp_answers[patient_idx].replace("PATIENT", "AGENT")
        new_meta.answers = temp_answers
        # reset this idxes
        core_idx = get_core_idxes_from_meta(new_meta)

    core_info = [
        ("AGENT", core_idx.aidx, core_changes.is_change_agent, core_changes.agent_changes),
        ("PATIENT", core_idx.pidx, core_changes.is_change_patient, core_changes.patient_changes),
    ]
    sorted_core_info = list(
        reversed(sorted(core_info, key=lambda tup: tup[1] if tup[1] is not None else 0))
    )
    # iterate in order of eidx decreasing so that deletion does not lead to wrong behavior
    for (core_tag, eidx, is_change, changes) in sorted_core_info:
        if is_change and eidx is not None:
            base = base_meta.core_args[eidx]
            if changes.is_change_content_type:
                # if target content type given, use; else, randomly sample other
                if changes.target_content_type:
                    new_type = changes.target_content_type
                else:
                    cand_types = [
                        x for x in ["sparse", "partial", "complete"] if x != base.tlemma_type
                    ]
                    new_type = np.random.choice(cand_types, 1)[0]
                    assert len(cand_types) == 2
                    assert f"{core_tag}+{new_type}" not in base_prompt
                new_meta.core_args[eidx].tlemma_type = new_type
            if changes.is_change_content:
                if changes.target_content:
                    new_content = changes.target_content
                    keyword_type = new_meta.core_args[eidx].tlemma_type
                else:
                    assert common_keywords_by_tag is not None, (
                        "If target keywords/content not supplied "
                        + f"and change {core_tag} content,"
                        + "need to provide common_keywords_by_tag, but got None"
                    )
                    # if content is RANDOM_TAG/*, cannot look-up by keyword type
                    keyword_type = (
                        new_meta.core_args[eidx].tlemma_type
                        if base.tlemma != RANDOM_TAG
                        else np.random.choice(["sparse", "partial", "complete"], 1)[0]
                    )
                    new_content = sample_common_keyword(
                        common_keywords_by_tag, base.tag, keyword_type, base.tlemma
                    )
                new_meta.core_args[eidx].tlemma = new_content
                new_meta.core_args[eidx].tlemma_type = keyword_type

            # TODO: make sure this is robust
            if changes.is_delete:
                # only delete if arg exists
                del new_meta.core_args[eidx]

    # filter out the random taged ones
    # new_meta.noncore_args = [a for a in new_meta.noncore_args if a.tlemma != RANDOM_TAG]
    # new_meta.core_args = [a for a in new_meta.noncore_args if a.tlemma != RANDOM_TAG]
    # core_idx = get_core_idxes_from_meta(new_meta)

    if noncore_changes.is_change_noncore:
        new_noncore_args = deepcopy(new_meta.noncore_args)
        new_answers = deepcopy(new_meta.answers)

        # get possible changes upfront to avoid re-computing
        role_dicts = get_possible_tags_by_vlemma(vlemma, frameset_path=frameset_path)
        if not role_dicts:
            warnings.warn(f"Returning None because could not get role dicts for verb {vlemma}.")
            return None

        # get possible tags for current frameset id, but if doesn't exist, get changes for all ids
        possible_changes = (
            list(set(role_dicts[base_meta.frameset_id].values()))
            if base_meta.frameset_id in role_dicts
            else list(
                set(
                    np.concatenate(
                        [list(frameset_dict.values()) for _, frameset_dict in role_dicts.items()]
                    )
                )
            )
        )
        unused_possible_changes = set(possible_changes)
        # modify every existing arg

        is_modify_all = any([x == "ALL" for x in noncore_changes.changes_by_arg.keys()])
        # exclude modals when modifying all because tied to tense
        tags_to_modify = (
            [x.tag for x in new_noncore_args if x.tag != "MODAL"]
            if is_modify_all
            else list(noncore_changes.changes_by_arg.keys())
        )

        # TODO: right now, can only properly handle one of same tag
        tags_to_modify = list(set(tags_to_modify))

        # expand noncore_changes dict to have content for each arg we are modifying
        if is_modify_all:
            noncore_changes.changes_by_arg = {
                k: noncore_changes.changes_by_arg["ALL"] for k in tags_to_modify
            }

        for orig_arg in tags_to_modify:
            arg_change_type = noncore_changes.changes_by_arg[orig_arg]
            # TODO: change to handle multiple disconnected spans of same arg
            # TODO: delete meta arg info associated with the target tag (don't want repeat modifiers)
            arg_idx = [idx for idx, tag in enumerate(new_noncore_args) if tag.tag == orig_arg]

            # if arg does not already exist, introduce it
            # TODO: do we always want this behavior?
            # some way of specifying in parse code whether to override nonexistent arg?
            if not arg_idx:
                arg_idx = None
                blank_idx = None
                orig_tag = orig_arg
                orig_tlemma = RANDOM_TAG
                orig_tlemma_type = None
            else:  # get original arg info
                arg_idx = arg_idx[0]
                orig_arg_info = new_meta.noncore_args[arg_idx]
                blank_idx = orig_arg_info.blank_idx
                orig_tag = orig_arg_info.tag
                assert orig_tag == orig_arg, f"{orig_tag} != {orig_arg}"
                orig_tlemma = orig_arg_info.tlemma
                orig_tlemma_type = orig_arg_info.tlemma_type

            if arg_change_type.is_delete:
                # only delete if arg exists
                if arg_idx is not None:
                    del new_noncore_args[arg_idx]
                new_meta.noncore_args = new_noncore_args
                new_meta.answers = new_answers
                continue

            if arg_change_type.is_change_tag:
                target_tag = arg_change_type["target_tag"]
                # if target_tag not supplied, randomly choose
                if not target_tag:
                    target_tag = np.random.choice(list(unused_possible_changes), 1, replace=False)[
                        0
                    ]
                    unused_possible_changes.remove(target_tag)
                if target_tag is None:
                    continue  # TODO: why would this happen?
            else:
                target_tag = orig_tag

            if arg_change_type.is_change_content_type:
                target_tlemma_type = arg_change_type["target_content_type"]
                # if target_tlemma_type not supplied, randomly choose
                if not target_tlemma_type:
                    cand_tlemma_types = [
                        x for x in ["sparse", "partial", "complete"] if x != orig_tlemma_type
                    ]
                    # TODO fix hacky: because negation spans are so short, NEGATION/SPARSE is empty
                    if target_tag == "NEGATION" and "sparse" in cand_tlemma_types:
                        cand_tlemma_types.remove("sparse")
                    target_tlemma_type = np.random.choice(cand_tlemma_types, 1)[0]
            else:
                target_tlemma_type = orig_tlemma_type

            if arg_change_type.is_change_content:
                if arg_change_type.target_content is None:
                    # if original tlemma is RANDOM_TAG, then corresponding tlemma_type is None;
                    # need to create new tlemma_type
                    # TODO: does this work when is_change_content AND is_change_content_type
                    # being used together? maybe redundant choosing of new target_tlemma_type?
                    if target_tlemma_type is None:
                        cand_tlemma_types = ["sparse", "partial", "complete"]
                        # TODO fix hacky: because negation spans are so short, NEGATION/SPARSE is empty
                        if target_tag == "NEGATION":
                            cand_tlemma_types.remove("sparse")
                        target_tlemma_type = np.random.choice(cand_tlemma_types, 1)[0]
                    assert common_keywords_by_tag is not None, (
                        "If target keywords/content is not supplied "
                        + "and change noncore arg content, need to provide common_keywords_by_tag,"
                        + "but got None"
                    )

                    tlemma = sample_common_keyword(
                        common_keywords_by_tag, target_tag, target_tlemma_type, orig_tlemma
                    )
                else:
                    tlemma = arg_change_type.target_content
            else:
                tlemma = orig_tlemma

            # TODO: setting raw_tag for arg to None; need to map to control code if we need raw_tag
            new_arg_info = Munch(
                raw_tag=None,
                tlemma=tlemma,
                tlemma_type=target_tlemma_type,
                tag=target_tag,
                blank_idx=blank_idx,
            )

            # add or modify existing meta non core info based on whether tag already exists
            # TODO: think more about how to encode answers; they do not match up to prompt as is,
            # cannot be used for UL training
            if arg_idx is not None and new_noncore_args[arg_idx].blank_idx is not None:
                new_noncore_args[arg_idx] = new_arg_info
                # update answers and noncore args for UL training (need corresponding 'neg' answers)
                # get answers idx of modified arg (indices of answers corresponds to blank indexes)
                answer_idx = new_meta.blank_indexes.index(new_noncore_args[arg_idx].blank_idx)
                # update control code of answers but not tlemma
                new_answers[answer_idx] = new_answers[answer_idx].replace(orig_arg, target_tag)
            else:
                new_noncore_args.append(new_arg_info)
                arg_idx = (
                    len(new_noncore_args) - 1
                )  # set for checking equality with arg_idx when deleting existing target tag

            # if changing tag, delete target tag if it already exists
            # TODO: is this behavior we want?
            if arg_change_type.is_change_tag:
                target_arg_idx = [
                    idx
                    for idx, tag in enumerate(new_noncore_args)
                    if tag.tag == target_tag and idx != arg_idx
                ]
                for idx in reversed(sorted(target_arg_idx)):
                    del new_noncore_args[idx]

            # if the sampled tag is the same as the original, continue
            if target_tag == orig_tag and arg_change_type.is_change_tag:
                continue

            new_meta.noncore_args = new_noncore_args
            new_meta.answers = new_answers

    # need to change when: vform changed and agent type originally complete,
    # agent/patient keywords are swapped with complete tag, etc.
    # based on verb voice (after changing vform), remove/add "by" from agent/patient keywords
    # (need to edit both agent/patient because some perturbation functions naively swap
    # TODO: make more robust

    def update_arg_keyword(eidx, keyword_convert_func):
        if eidx is not None:
            arg = new_meta.core_args[eidx]
            if arg.tlemma is not None and arg.tlemma_type == "complete":
                new_meta.core_args[eidx].tlemma = keyword_convert_func(arg.tlemma)

    # TODO: not going to work when patient or agent starts with name
    def swap_capitalizations(core_idx):
        patient_arg = new_meta.core_args[core_idx.pidx] if core_idx.pidx is not None else None
        agent_arg = new_meta.core_args[core_idx.aidx] if core_idx.aidx is not None else None
        if patient_arg is not None and agent_arg is not None:
            patient_kw = patient_arg.tlemma
            agent_kw = agent_arg.tlemma
            patient_is_upper = patient_kw[0].isupper()
            agent_is_upper = agent_kw[0].isupper()
            if patient_is_upper:
                new_meta.core_args[core_idx.aidx].tlemma = uppercase_keyword(agent_kw)
            else:
                new_meta.core_args[core_idx.aidx].tlemma = lowercase_keyword(agent_kw)

            if agent_is_upper:
                new_meta.core_args[core_idx.pidx].tlemma = uppercase_keyword(patient_kw)
            else:
                new_meta.core_args[core_idx.pidx].tlemma = lowercase_keyword(patient_kw)

    # get new core idxes because might've changed (after deleting)
    core_idx = get_core_idxes_from_meta(new_meta)
    if new_meta.vvoice == "passive" and verb_changes.is_change_vvoice:
        update_arg_keyword(core_idx.aidx, passivize_keyword)
        update_arg_keyword(core_idx.pidx, unpassivize_keyword)
    elif new_meta.vvoice == "active" and verb_changes.is_change_vvoice:
        update_arg_keyword(core_idx.aidx, unpassivize_keyword)
        update_arg_keyword(core_idx.pidx, unpassivize_keyword)

    # swap capitalizations because we expect kws to move
    if verb_changes.is_change_vvoice:
        swap_capitalizations(core_idx)

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
            # TODO: make more robust; now, if is_delete_text, delete all words that are
            # not in string punctuation
            if context_changes.is_delete_punct and word in string.punctuation:
                continue
            if context_changes.is_delete_text and word not in string.punctuation:
                continue
            new_words.append(word)
        new_nonheader = " ".join(new_words)
        new_prompt = header + " " + new_nonheader

    # TODO: do we want warning behavior? or an error? or return None?
    if new_prompt == base_prompt:
        warnings.warn(
            "Warning. Perturbed prompt is same as base prompt. This should not be happening."
        )
        return None

    new_prompt = Munch(
        prompt=new_prompt, perturb_meta=perturb_meta, meta=new_meta, answer=new_answer
    )
    return new_prompt


######################################################################
# parse and recover the prompt/fill in
######################################################################


class BadGenerationError(Exception):
    pass


def clean_prefixes_in_filled_prompt(filled_prompt):
    try:
        header, _ = extract_header_from_prompt(filled_prompt)
        filled_prompt = filled_prompt.replace(header, "").strip()
    except RuntimeError:
        pass
    regex = re.compile(r"\[FILL[0-9] \| (?P<fillin>[^\]]*)\]")
    filled_prompt = regex.sub(r"\1", filled_prompt)
    regex = re.compile(r"\[[A-Z\s\+]+: (?P<fillin>[^\]]*)\]")
    filled_prompt = regex.sub(r"\1", filled_prompt)
    regex = re.compile(r"\[VERB \| (?P<fillin>[^\]]*)\]")
    filled_prompt = regex.sub(r"\1", filled_prompt).split("]", 1)[-1].strip()
    return re.sub(" +", " ", filled_prompt)


def is_bad_generation(gen):
    return "sanatate" in gen


# fill in and parse the prompt
def fillin_prompt(prompt, generated_text, is_clean_prefix=False):
    """Fill in the prompt
    Args:
        prompt (str): the input prompt, with head (see output of `gen_header_by_meta`)
            and the blanked sentence
        generated_text (str): the output of the t5 perturber.
        is_clean_prefix (bool): whether to add prefix.
    Returns:
        str: the filled in prompt.
            If cannot correctly fill in, return None.
            All the filled in has a wrapper of [FILL{idx} | {text}] so it can be
            parsed by `parse_filled_prompt` -- to identify places being changed.
    """
    generated_text = generated_text.replace("</s>", "").replace("<pad>", "").replace("<unk>", "")
    if is_bad_generation(generated_text):
        raise BadGenerationError("Bad generation: {}".format(prompt))
    header, _ = extract_header_from_prompt(prompt)
    if header:
        prompt_fix = header.strip() + " " + generated_text
    else:
        prompt_fix = generated_text
    if is_clean_prefix:
        prompt_fix = clean_prefixes_in_filled_prompt(prompt_fix)
    prompt_fix = re.sub(r"<extra_id_[0-9]>", " ", prompt_fix)
    return re.sub(" +", " ", prompt_fix)


def parse_filled_prompt(
    prompt,
    doc=None,
    nlp=None,
    is_compute_vidx=True,
    is_most_likely_tag=True,
    is_include_raw_tag=False,
):
    # TODO: have annotations' tags be *control codes/raw tags* of generations
    """Fill in the prompt
    Args:
        prompt (str): the filled prompt, see the output of `fillin_prompt`
        doc (Doc, Optional): the original doc of the sentence.
            Defaults to None, but cannot be None together with nlp.
        nlp : spacy processor, for generating doc if doc is none
        is_compute_vidx (bool, Optional): whether to compute the verb index.
            This is because the verb chunk index can be of the aux verb;
            If so, it's important to get back to the core idx, or the prediction
            matching wouldnt work.
            If the goal is only to compute the sentence, don't need to correctly
            recover the idx.
    Returns:
        Munch: {
            prompt: str, the input string repeated.
            sentence: str, the natural language sentence.
            meta: the meta info of the control, see output to `extract_meta_from_prompt`
            annotations: Munch[], the identified spans being changed, and the verb.
                [{tag: 'VERB', star: 6, end: 7}, {tag: 'FILL1', start: 0, end: 1}],
            words: str[], the tokenized words of sentence,
            vidx: int, verb index
        }
    """
    if not prompt:
        return {}
    try:
        meta, header = extract_meta_from_prompt(prompt, return_header=True)
    except Exception:
        # redirect caught exception to BadGenerationError
        raise BadGenerationError(f"Bad generation: {prompt}") from None
    if header:
        # remove the header
        prompt = prompt.replace(header, "").strip()
    else:
        # TODO: hacky, if extract_meta_from_prompt returns None, None,
        # it's because header is not part of prompt
        # (i.e. when generator outputs whole sequence so filled prompt does not contain header)
        prompt = prompt
    words = []
    annotations = []
    # get the verbs
    # TODO: check robustness of re expressions
    # TODO: allow tag spans to be enclosed in quotes?
    additional_tag = re.escape("-") if is_include_raw_tag else ""
    tags = []
    for idx, exp in enumerate(
        [
            r"\[+(?P<tag>[a-zA-Z0-9{}\s]+)(?P<punct>[{}\s]+)(?P<fillin>[^\]\[]*)\]+".format(
                additional_tag, re.escape(string.punctuation)
            ),
            # at least match one side of []
            r"\[+(?P<tag>[A-Z0-9{}]+)(?P<punct>[{}\s]+)(?P<fillin>[^\]\[]*)\]*".format(
                additional_tag, re.escape(string.punctuation)
            ),
            r"\[*(?P<tag>[A-Z0-9{}]+)(?P<punct>[{}\s]+)(?P<fillin>[^\]\[]*)\]+".format(
                additional_tag, re.escape(string.punctuation)
            ),
        ]
    ):
        # try several iterations of the tags
        if idx == 0:
            tags += list(re.finditer(exp, prompt))
        else:
            for m in re.finditer(exp, prompt):
                # not already matched by the first one
                if all([(tag.end(0) <= m.start(0)) or (m.end(0) <= tag.start(0)) for tag in tags]):
                    tags.append(m)
    vidx = -1
    gold_tags = get_argm_and_core_values()
    prev_end = 0
    tags = sorted(tags, key=lambda t: t.start(0))
    for m in tags:
        raw_tag, _, fillin = m.group("tag"), m.group("punct"), m.group("fillin")
        curr_words = []
        prev = prompt[prev_end : m.start(0)]
        tag = find_most_likely_tag(raw_tag, gold_tags) if is_most_likely_tag else raw_tag
        prev_end = m.end(0)
        tag = tag.replace(" ", "")
        prev = re.sub(" +", " ", prev.strip())
        # text = re.sub(' +', ' ', text.strip())
        fillin = re.sub(" +", " ", fillin.strip())
        if prev:
            if nlp:
                curr_tokens = [t.text for t in nlp.tokenizer(prev)]
            else:
                curr_tokens = prev.split(" ")
            words += [w for w in curr_tokens if w]
        if fillin:
            if nlp:
                curr_words = [t.text for t in nlp.tokenizer(fillin)]
            else:
                curr_words = fillin.split(" ")
            # words += [w for w in curr_tokens if w]
        annotations.append(
            Munch(tag=tag, start=len(words), end=len(words) + len(curr_words), pred="")
        )
        if tag == "VERB" or tag == "V":
            vidx = len(words)
        words += curr_words
    words += [w for w in prompt[prev_end:].strip().split(" ") if w]
    sentence = " ".join(words)
    if (doc or nlp) and is_compute_vidx:
        if not doc:
            doc = nlp(sentence)
        while vidx >= 0 and vidx < len(doc) and doc[vidx].dep_.startswith("aux"):
            vidx = doc[vidx].head.i
    sentence = re.sub(" +", " ", " ".join(words).strip())
    # do a final check on the parse:
    if any(
        [
            punct0 + g + punct1 in sentence
            for punct0, g, punct1 in itertools.product(
                string.punctuation + " ", get_argm_and_core_values(), string.punctuation
            )
        ]
    ):
        sentence = None
    # exp = r'(?P<tag>[{}\s]+)(?P<punct>[{}\s]+)'.format(
    #    "|".join(get_argm_and_core_values()), re.escape(string.punctuation))
    # if does not match, return None
    # if re.match(exp, sentence): sentence = None
    if sentence is None or len(annotations) == 0:
        raise BadGenerationError(f"Bad generation: {prompt}")

    clean_sentence = clean_punct(sentence)

    return Munch(
        prompt_no_header=prompt,
        sentence=sentence,
        clean_sentence = clean_sentence,
        meta=meta,
        words=words,
        vidx=vidx,
        is_valid=True,
        annotations=annotations,
    )


def add_predictions_to_prompt_dict(generated_dict, predicted, frameset_path=DEFAULT_FRAME_SET_PATH):
    # TODO: update this function with new prompt format
    """Add the pred dict info to the prompt dict parsed by `parse_filled_prompt`
    Args:
        generated_dict (Munch): Output of `parse_filled_prompt`
        predicted (Dict): the predicted tags for the sentence
            {
                'verbs': [{'verb': str, "description": str, "tags": str[]},
                "words": str[]
            }
    Returns:
        Munch: still the prompt dict, just with the pred info updated
    """

    # this part adds the prediction to the prompts
    if not len(predicted["words"]) == len(generated_dict.sentence.split(" ")):
        generated_dict.is_valid = False
        return
    vidx = generated_dict.vidx
    vlemma = generated_dict.meta.vlemma
    try:
        frameset_id = generated_dict.meta.frameset_id
    except AttributeError:
        frameset_id = "01"
    pred_dict = {get_vindex_by_tags(v["tags"]): deepcopy(v["tags"]) for v in predicted["verbs"]}
    if vidx not in pred_dict:
        generated_dict.is_valid = False
        return generated_dict
    raw_tags = pred_dict[vidx]
    for ann in generated_dict.annotations:
        pred = get_tag_for_span(
            raw_tags, frameset_id, ann.start, ann.end, vlemma=vlemma, frameset_path=frameset_path
        )
        if pred:
            label = convert_tag2readable(vlemma, pred, frameset_id, frameset_path=frameset_path)
            ann.pred = label if label else pred
    return generated_dict


def add_predictions_to_prompt_dict_new(
    generated_dict, predicted, frameset_path=DEFAULT_FRAME_SET_PATH
):
    # TODO: update this function with new prompt format
    """Add the pred dict info to the prompt dict parsed by `parse_filled_prompt`
    Args:
        generated_dict (Munch): Output of `parse_filled_prompt`
        predicted (Dict): the predicted tags for the sentence
            {
                'verbs': [{'verb': str, "description": str, "tags": str[]},
                "words": str[]
            }
    Returns:
        Munch: still the prompt dict, just with the pred info updated
    """

    # this part adds the prediction to the prompts
    if not len(predicted["words"]) == len(generated_dict.sentence.split(" ")):
        print("Length mismatch!!")
        generated_dict._replace(is_valid=False)
        return
    vidx = generated_dict.vidx
    vlemma = generated_dict.meta.vlemma
    try:
        frameset_id = generated_dict.meta.frameset_id
    except AttributeError:
        frameset_id = "01"
    pred_dict = {get_vindex_by_tags(v["tags"]): deepcopy(v["tags"]) for v in predicted["verbs"]}
    if vidx not in pred_dict:
        print("Missing vidx!!")
        print("vidx", vidx)
        print("pred_dict", pred_dict)
        generated_dict._replace(is_valid=False)
        return generated_dict
    raw_tags = pred_dict[vidx]
    for ann in generated_dict.annotations:
        pred = get_tag_for_span(
            raw_tags, frameset_id, ann.start, ann.end, vlemma=vlemma, frameset_path=frameset_path
        )
        if pred:
            label = convert_tag2readable(vlemma, pred, frameset_id, frameset_path=frameset_path)
            ann.pred = label if label else pred
    return generated_dict


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
    except (ValueError, IndexError):
        original_blank_idx = None
    # can only shorten an existing argument
    if original_blank_idx is None:
        return None

    blank_idx = meta.blank_indexes[original_blank_idx]
    return meta.doc[blank_idx[0] : blank_idx[1]]


def capitalize_by_voice(verb_voice, agent_kw, patient_kw):
    """Helper function to fix capitalizations of agent/patient keywords based on verb voice
    This is to encourage generating full sentence without hallucinating context,
    since generator is case sensitive.
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
        warnings_message = (
            "Trying to change capitalization of agent and patient keywords, "
            "but got None for both arguments"
        )
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
