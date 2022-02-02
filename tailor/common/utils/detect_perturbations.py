import os
from collections import Counter

import json
from munch import Munch
from copy import deepcopy
from tailor.common.utils.head_prompt_utils import (
    get_vindex_by_tags,
    clean_prefix_for_one_tag,
    gen_prompts_by_tags,
    convert_tag2readable,
    gen_prompt_by_perturb_meta,
    is_equal_prompts,
    parse_change_type_meta,
    parse_keyword_type,
    get_keyword_candidates_for_span,
    parse_filled_prompt,
    get_unique_tags,
)

from tailor.common.utils.tag_utils import DEFAULT_FRAME_SET_PATH
from tqdm import tqdm

# DEFAULT_DATASET_PATH = "../../label-contrast-generation/label_contrast/srl/data/orig/train.json"

DEFAULT_DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "srl_orig_train.json")
)

DEFAULT_COMMON_KEYWORDS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "common_keywords_by_tag.json")
)


def parse_base_props(prop, nlp, frameset_path):
    try:
        train_dict = parse_filled_prompt(
            prop["description"], nlp=nlp, is_most_likely_tag=False, is_include_raw_tag=True
        )
        vlemma, frameset_id = prop["lemma"], prop["frameset_id"]
    except Exception as e:
        return {}
    if "annotations" not in train_dict:
        return train_dict
    for ann in train_dict.annotations:
        tag = convert_tag2readable(vlemma, ann.tag, frameset_id, frameset_path=frameset_path)
        if tag:
            ann.tag = tag
    return train_dict


def get_common_keywords_by_tag(
    data_path=DEFAULT_DATASET_PATH,
    keyword_str="NOUN_CHUNKS,PREFIX,CONNECT,ROOT",
    nlp=None,
    frameset_path=DEFAULT_FRAME_SET_PATH,
    max_count=100,
):
    """Counts all keyword candidates that have a type specified by keyword_str
        Used for UL training for keyword following
        Helper for sample_common_keyword()
        Gets dict of common keywords by tag
    Args:
        raw_json: list of jsons representing original data
        keyword_str: str to use in getting keyword candidates
    Returns:
        Dict of Dicts, where each outer dictionary has
            key: SRL control code
            value: sub-dictionary with
                sub-key: one of complete/partial/sparse (specifying the keyword type)
                sub-value: collections.Counter object (specifying frequency of different keywords)
    """
    assert nlp is not None
    train_dicts = []
    class_labels = set()
    vlemmas = []

    with open(data_path, "r") as f:
        raw_json = json.load(f)
    raw_json = [r for r in raw_json if len(r["props"])]
    if max_count is not None:
        raw_json = raw_json[:max_count]
    for idx, train in tqdm(enumerate(raw_json), total=len(raw_json)):
        for prop in train["props"]:
            try:
                train_dict = parse_base_props(prop, nlp=nlp, frameset_path=frameset_path)
                class_labels.update(set([ann.tag for ann in train_dict.annotations]))
                train_dicts.append(train_dict)
                vlemmas.append(prop["lemma"])
            except Exception as e:
                continue
    all_s = set([s.sentence for s in train_dicts])
    processed_s = list(nlp.pipe(all_s))
    process_dict = dict(list(zip(all_s, processed_s)))

    keyword_types = ["complete", "partial", "sparse"]
    key_dicts = {}
    for label in class_labels:
        # use vlemma in original data for VERB data
        if label == "VERB":
            continue
        key_dicts[label] = {k: [] for k in keyword_types}
        for train_dict in train_dicts:
            doc = process_dict[train_dict["sentence"]]
            for ann in train_dict.annotations:
                if ann.tag == label:
                    keyword_meta = parse_keyword_type(keyword_str)
                    keywords = get_keyword_candidates_for_span(
                        doc[ann.start : ann.end], keyword_meta
                    )
                    temp_dict = {k: [] for k in keyword_types}
                    for keyword_type, keyword in keywords:
                        if keyword_type not in temp_dict:
                            temp_dict[keyword_type] = []
                        temp_dict[keyword_type].append(keyword)
                    for keyword_type in keyword_types:
                        key_dicts[label][keyword_type].extend(temp_dict[keyword_type])
    key_dicts["VERB"] = {"complete": vlemmas, "sparse": [], "partial": []}

    for label in class_labels:
        for keyword_type in keyword_types:
            key_dicts[label][keyword_type] = Counter(key_dicts[label][keyword_type])
    return key_dicts


def identify_tags_for_span(doc, predicted, start=None, end=None):
    """Identify possible SRL tags and their corresponding VERB, given a span.
    The should be used for suggestions on possible changes of a span.
    Args:
        span (Span): spacy span
        predicted (List): the predicted tags for the sentence
            [{'verb': str, "description": str, "tags": str[]}, {...}]
    Returns:
        Munch[]: A list of possible
            {
                'tags': str[],
                'span_raw_tag': str,
                'start': int, 'end': start,
                'is_core': bool, whether the span is related to core arguments
                'vlemma': str
            }
    """

    start = start if start is not None and start > 0 else 0
    end = end if end is not None and start <= len(doc) else len(doc)
    # assert the length of tokenization

    # if not len(predicted["words"]) == len(doc):
    #     return []
    # iterate through each of the verb case
    possible_tags = []
    pred_dict = {get_vindex_by_tags(v["tags"]): deepcopy(v["tags"]) for v in predicted}
    for vidx, raw_tags in pred_dict.items():
        # verb idx
        clean_tags = [clean_prefix_for_one_tag(p) for p in raw_tags]
        local_preds = set([p for p in clean_tags[start:end] if p not in ["O"]])
        vlemma = doc[vidx].lemma_
        # all possible changes
        for pred in local_preds:
            # translate to human readable case
            # label = search_label_for_tag(vlemma, pred)
            # if label is None:
            #    continue
            # fix the indexes
            local_start = clean_tags.index(pred, start, end)
            local_end = local_start + 1
            while local_start > 0 and clean_tags[local_start - 1] == pred:
                local_start -= 1
            while local_end < len(clean_tags) - 1 and clean_tags[local_end] == pred:
                local_end += 1
            span_type = "noncore"
            if pred == "ARG0" or pred == "ARG1":
                span_type = "core"
            elif pred == "V":
                span_type = "verb"
            possible_tags.append(
                Munch(
                    raw_tags=raw_tags,
                    vlemma=vlemma,
                    span_raw_tag=pred,
                    start=local_start,
                    end=local_end,
                    span_type=span_type,
                )
            )
    if not possible_tags:
        # find the verb and then default to core arg
        root = doc[start:end].root
        while root.i not in pred_dict:
            root = root.head
        if root.i in pred_dict:
            possible_tags.append(
                Munch(
                    raw_tags=pred_dict[root.i],
                    vlemma=root.lemma_,
                    span_raw_tag="O",
                    start=start,
                    end=end,
                    span_type=None,
                )
            )
    return possible_tags


def detect_perturbations(
    doc, start, end, predicted, frameset_path=DEFAULT_FRAME_SET_PATH, common_keywords_by_tag=None
):
    candidates = identify_tags_for_span(doc, predicted, start, end)
    output = []
    for candidate in candidates:
        if candidate.span_type is None:
            continue
        unique_tags = get_unique_tags(candidate.raw_tags)

        tag = convert_tag2readable(
            candidate.vlemma, candidate.span_raw_tag, None, frameset_path=frameset_path
        )
        perturb_strs = []
        if candidate.span_type == "core":
            perturb_strs = [
                (f"swap_core", "CORE(SWAP_CORE)", "NOUN_CHUNKS,RANDOM_SUBTREES"),
                (
                    f"change_content",
                    f"CORE({tag}:CHANGE_CONTENT)",
                    "NOUN_CHUNKS,PREFIX,CONNECT,ROOT",
                ),
                (f"change_spec", f"NONCORE({tag}:CHANGE_SPECIFICITY(sparse))", "EXACT,UNCASED"),
            ]
        elif candidate.span_type == "noncore":
            perturb_strs = [
                (f"move", "CONTEXT(CHANGE_IDX(0:-1))", "EXACT,UNCASED"),
                (f"move", "CONTEXT(CHANGE_IDX(-1:0))", "EXACT,UNCASED"),
                (
                    f"change_content",
                    f"NONCORE({tag}:CHANGE_CONTENT)",
                    "NOUN_CHUNKS,PREFIX,CONNECT,ROOT",
                ),
                (f"change_spec", f"NONCORE({tag}:CHANGE_SPECIFICITY(sparse))", "EXACT,UNCASED"),
                (f"change_tag", f"NONCORE({tag}:CHANGE_TAG)", "NOUN_CHUNKS,PREFIX,CONNECT,ROOT"),
            ]
        elif candidate.span_type == "verb":
            perturb_strs = [
                (f"change_tense", f"VERB(CHANGE_TENSE)", "EXACT,UNCASED"),
                (f"change_voice", f"VERB(CHANGE_VOICE)", "NOUN_CHUNKS,RANDOM_SUBTREES"),
                (f"change_content", f"VERB(CHANGE_VLEMMA)", "EXACT,UNCASED"),
            ]
        for name, perturb_str, keyword_str in perturb_strs:
            if candidate.start == 0 and "front" in name:
                continue
            if candidate.end == len(doc) and "back" in name:
                continue
            # if the span is already too long, skip
            if candidate.end - candidate.start > 5 and "spec" in name:
                continue
            args_to_blank = [candidate.span_raw_tag]
            if "V" not in args_to_blank:
                args_to_blank.append("V")
            if name == "swap_core" or name == "change_voice":
                args_to_blank = [f for f in ["ARG0", "ARG1", "V"] if f in unique_tags]
            orig_prompt = gen_prompts_by_tags(
                doc,
                None,
                candidate.raw_tags,
                return_prompt_type="concrete",
                nblanks=1,
                args_to_blank=args_to_blank,
                keyword_str=keyword_str,
                frameset_path=frameset_path,
            )
            # core_idx = get_core_idxes_from_meta(case.meta)

            perturb_meta = parse_change_type_meta(perturb_str)
            perturbed = gen_prompt_by_perturb_meta(
                doc,
                candidate.raw_tags,
                perturb_meta,
                orig_prompt.meta,
                frameset_path=frameset_path,
                common_keywords_by_tag=common_keywords_by_tag,
            )
            if perturbed is None:
                continue
            # print("| ORIG: ", orig_prompt.prompt)
            # print("| EDIT: ", perturbed.prompt)
            # print()
            if not is_equal_prompts(perturbed.prompt, orig_prompt.prompt):
                output.append(Munch(constraint_type=name, prompt=perturbed.prompt))
    return output
