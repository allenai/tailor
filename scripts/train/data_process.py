import os
import sys
sys.path.append('../..')
sys.path.append('..')

import json
import spacy
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from collections import Counter
#from label_contrast.srl.analysis_utils import feature_extract
import itertools

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
#from nltk import everygrams
from scipy.sparse import find

import functools
import logging
from munch import Munch

#from tailor.common.utils.head_prompt_utils import *
#from tailor.common.utils.model_utils import *
from tailor.common.utils.tag_utils import DEFAULT_FRAME_SET_PATH
from tailor.common.utils import get_spacy_model
from tailor.common.utils.perturbation_controls import parse_change_type_meta
from tailor.common.utils.head_prompt_utils import (
    gen_random_prompt,
    get_core_idxes_from_meta,
    get_keyword_candidates_for_span,
    parse_keyword_type,
    RANDOM_TAG,
    gen_prompt_by_perturb_meta,
    parse_filled_prompt,
    convert_tag2readable,
)

######################################################################
# for parsing the conll code
######################################################################

logger = logging.getLogger(__name__)

def convert_jsons(domain, words, props, tags, lemmas, frameset_ids):
    jsons = []
    def _convert_desc(words, tags, vidx):
        curr_tags = tags[vidx][:]
        output = []
        t, raw_tag = "", ""
        for idx, (tag, word) in enumerate(zip(curr_tags, words)):
            if t != "" and not tag.startswith("I-"):
                output.append(f"[{raw_tag}: {t}]")
                t, raw_tag = "", ""
            
            if tag.startswith("B-"): 
                t, raw_tag = word, tag[2:]
            elif tag.startswith("I-"): 
                t += " " + word
            else:
                output.append(word)
        return " ".join(output)
    propid_labels = ['O' for _ in words]
  
    lemmas = [x for x in lemmas if x != "-"]
    frameset_ids = [x for x in frameset_ids if x != "-"]
    assert len(lemmas) == len(props)
    assert len(lemmas) == len(frameset_ids)

    if len(words) != 0:
        sentence = {
            "domain": domain,
            "words": words,
            "lemmas": lemmas,
            "frameset_ids": frameset_ids,
            "props": [],
        }
        for vidx in range(len(props)):
            assert len(tags[vidx]) == len(words)
            assert tags[vidx][props[vidx]] in {"B-V", "B-I"}
            sentence["props"].append({
                "vidx": vidx,
                "tags": tags[vidx], 
                "description": _convert_desc(words, tags, vidx),
                "lemma": lemmas[vidx],
                "frameset_id": frameset_ids[vidx],
            })
        jsons.append(sentence)
    return jsons


def conlldoc2jsons(domain, ctext):
    jsons = []
    words, props, tags, spans, all_props, lemmas, frameset_ids = [], [], [], [], [], [], []
    for line in ctext.split("\n"):
        line = line.strip()
        if line == '': # finished one instance
            joined_words = " ".join(words)
            prev_words = joined_words
            jsons += convert_jsons(domain, words, props, tags, lemmas, frameset_ids)
            words, props, tags, spans, all_props, lemmas, frameset_ids = [], [], [], [], [], [], []
            continue
        if line[0] == "#": # starting or stopping a new instance
            prev_words = ""
            if len(words) > 0:
                jsons += convert_jsons(domain, words, props, tags, lemmas, frameset_ids)
            continue
        info = line.split()
        try:
            word = info[3]
        except UnicodeEncodeError:
            print (root, dirs, f)
            print (info[3])
            word = "*UNKNOWN*"
        words.append(word)
        idx = len(words) - 1
        if idx == 0: # starting word
            # create the matrix/default value for tags and spans
            tags = [[] for _ in info[11:-1]]
            spans = ["" for _ in info[11:-1]]
        is_predicate = (info[7] != '-')
        is_verbal_predicate = False
        lemma = info[6] if info[7] != '-' else '-'
        frameset_id = info[7]
        lemmas.append(lemma)
        frameset_ids.append(frameset_id)
        for t in range(len(tags)):
            arg = info[11 + t]
            label = arg.strip("()*")
            # split the tags
            if "(" in arg:
                tags[t].append("B-" + label)
                spans[t] = label
            elif spans[t] != "":
                tags[t].append("I-" + spans[t])
            else:
                tags[t].append("O")
            if ")" in arg:
                spans[t] = ""
            if "(V" in arg:
                is_verbal_predicate = True
        if is_verbal_predicate:
            props.append(idx)
    # final sentence
    jsons += convert_jsons(domain, words, props, tags, lemmas, frameset_ids)

    return jsons


def convert_all_conll_files(indir, outdir):
    jsons = []
    logger.info(f"| Reading data from dir: {indir}")
    for root, dirs, files in tqdm(os.walk(indir)):
        for f in files:
            if not 'gold_conll' in f:
                continue
            #print root, dirs, f
            dpath = root.split('/')
            domain = '_'.join(dpath[dpath.index('annotations')+1:-1])
            with open(f"{root}/{f}", "r") as f:
                ctext = f.read()
            jsons += conlldoc2jsons(domain, ctext)
    assert len(jsons) != 0
    logger.info(f"| Writing data to: {outdir}")
    json.dump(jsons, open(outdir, "w"))
    return jsons


def parse_base_props(prop, nlp, frameset_path):
    try:
        train_dict = parse_filled_prompt(
            prop["description"], nlp=nlp, is_most_likely_tag=False, is_include_raw_tag=True)
        vlemma, frameset_id = prop["lemma"], prop["frameset_id"]
    except Exception as e:
        print(e)
        return {}
    if "annotations" not in train_dict:
        return train_dict
    for ann in train_dict.annotations:
        tag = convert_tag2readable(
            vlemma, ann.tag, frameset_id, frameset_path=frameset_path)
        if tag:
            ann.tag = tag
    return train_dict

def get_common_keywords_by_tag(raw_json, keyword_str, 
        keyword_types = ["complete", "partial", "sparse"],
        frameset_path=DEFAULT_FRAME_SET_PATH):
    """ Counts all keyword candidates that have a type specified by keyword_str
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

    #nlp = get_spacy_model("en_core_web_sm") # TODO: debug; this does not work, reloading nlp for now
    nlp = spacy.load("en_core_web_sm")
    train_dicts = []
    class_labels = set()
    vlemmas = []

    logger.info("Getting common keywords (for UL training)...")

    for idx, train in tqdm(enumerate(raw_json), total=len(raw_json)):
        for prop in train["props"]:
            try:
                train_dict = parse_base_props(
                    prop, nlp=nlp, frameset_path=frameset_path)
                class_labels.update(set([ann.tag for ann in train_dict.annotations]))
                train_dicts.append(train_dict)
                vlemmas.append(prop["lemma"])
            except Exception as e:
                logger.info(f"Error parsing prop; error message: '{e}'; prop: {prop}")
                continue
    all_s = set([s.sentence for s in train_dicts])
    processed_s = list(nlp.pipe(all_s))
    process_dict = dict(list(zip(all_s, processed_s)))

    key_dicts = {}
    for label in class_labels:
        # use vlemma in original data for VERB data
        if label == "VERB":
            continue
        key_dicts[label] = {k:[] for k in keyword_types}
        for train_dict in train_dicts:
            doc = process_dict[train_dict["sentence"]]
            for ann in train_dict.annotations:
                if ann.tag == label:
                    keyword_meta = parse_keyword_type(keyword_str)
                    keywords = get_keyword_candidates_for_span(doc[ann.start:ann.end], keyword_meta)
                    temp_dict = {k:[] for k in keyword_types}
                    for keyword_type, keyword in keywords:
                        if keyword_type not in temp_dict:
                            temp_dict[keyword_type] = []
                        temp_dict[keyword_type].append(keyword)
                    for keyword_type in keyword_types:
                        key_dicts[label][keyword_type].extend(temp_dict[keyword_type])
    key_dicts["VERB"] = {"complete": vlemmas}
    for keyword_type in keyword_types:
        if keyword_type not in key_dicts["VERB"]:
            key_dicts["VERB"][keyword_type] = []

    for label in class_labels:
        for keyword_type in keyword_types:
            key_dicts[label][keyword_type] = Counter(key_dicts[label][keyword_type])
            logger.info(f"10 most common keywords for '{label}', {keyword_type}':")
            logger.info(f"{key_dicts[label][keyword_type].most_common(10)}")

    return key_dicts

######################################################################
# generate multiple cases per prompt
######################################################################


def get_head_prompts_from_json(
    raw_json, nlp, use_unlikelihood=False, 
    use_resampling=False, 
    n=None, return_sequence=False, 
    frameset_path=None,
    # some inputs to compute_weights
    is_resample_by_label=True, is_resample_by_feature=True, 
    is_ngrams_as_features=True,
    min_ngram=1, max_ngram=3, is_return_pre_weight_prompts=False, verbose=False,
    is_return_base_dict=False):
    
    raw_json = [r for r in raw_json if len(r["props"])]
    if n: raw_json = raw_json[:n]
    # save the spacy data
    all_s = set([str(" ".join(raw["words"]).strip()) for raw in raw_json])
    processed_s = list(nlp.pipe(all_s))
    process_dict = dict(list(zip(all_s, processed_s)))

    keyword_str = "NOUN_CHUNKS,RANDOM_SUBTREES,EXACT,PREFIX,CONNECT,ROOT"
    if use_unlikelihood:
        common_keywords_by_tag = get_common_keywords_by_tag(raw_json, keyword_str, frameset_path=frameset_path) 
    else:
        common_keywords_by_tag = {}
    num_subset, num_total, num_skipped = 0, 0, 0
    all_prompts = []
    for instance_idx, raw in tqdm(enumerate(raw_json), total=len(raw_json)):
        doc = process_dict[" ".join(raw["words"]).strip()]
        cases_by_curr_doc = {}
        assert len(raw['props']) == len(raw['frameset_ids'])
        assert len(raw['props']) == len(raw['lemmas'])
        for prop, frameset_id, vlemma in zip(raw["props"], raw["frameset_ids"], raw["lemmas"]):
            neg_cases_by_curr_tag = []
            raw_tags = prop["tags"]
            if is_return_base_dict:
                base_dict = parse_base_props(prop, nlp=nlp, frameset_path=frameset_path)
            else:
                base_dict = None
            do_print = True if verbose and (instance_idx < 10 or instance_idx % 500 == 0 or any(["B-R" in t or "B-C-" in t for t in raw_tags])) else False

            if any(["B-C-" in t for t in raw_tags]):
                num_skipped += 1 # TODO: handle B-R args
                continue
            #if True:
            try:
                case = gen_random_prompt(doc, frameset_id, raw_tags, 
                    return_sequence=return_sequence, vlemma=vlemma,
                    keyword_str=keyword_str,
                    return_prompt_type="all", frameset_path=frameset_path)
                if case is None:
                    continue
                cases_by_curr_doc[case.prompt] = True
                case.reward = 1
                case.instance_idx = instance_idx
                case.base_dict = base_dict
                meta = case.meta
                cases_by_curr_tag = [case]
                
                num_subset += int(case.is_subset)
                num_total += 1
                if use_unlikelihood:
                    tag_perturb_str = "VERB(CHANGE_TENSE, CHANGE_VOICE);CORE(SWAP_CORE);NONCORE(ALL:CHANGE_TAG)"
                    keyword_perturb_str = "VERB(CHANGE_VLEMMA);CORE(AGENT:CHANGE_CONTENT|PATIENT:CHANGE_CONTENT)" + \
                                        "NONCORE(ALL:CHANGE_CONTENT)"
                    specificity_perturb_str = "CORE(AGENT:CHANGE_SPECIFICITY|PATIENT:CHANGE_SPECIFICITY)" 

                    # if both agent and patient are random tag, cannot perturb specificity of keyword
                    core_idx = get_core_idxes_from_meta(case.meta)
                    if core_idx.aidx is not None and \
                        core_idx.pidx is not None and \
                        case.meta.core_args[core_idx.aidx].tlemma == RANDOM_TAG and \
                        case.meta.core_args[core_idx.pidx].tlemma == RANDOM_TAG:
                        perturb_strings = [tag_perturb_str, keyword_perturb_str]
                    else:
                        if core_idx.aidx is not None and case.meta.core_args[core_idx.aidx].tlemma == RANDOM_TAG:
                            specificity_perturb_str = specificity_perturb_str.replace("AGENT:CHANGE_SPECIFICITY", "")
                        if core_idx.pidx is not None and case.meta.core_args[core_idx.pidx].tlemma == RANDOM_TAG:
                            specificity_perturb_str = specificity_perturb_str.replace("PATIENT:CHANGE_SPECIFICITY", "")
                        perturb_strings = [tag_perturb_str, keyword_perturb_str, specificity_perturb_str]

                    for perturb_str in perturb_strings:
                        perturb_meta = parse_change_type_meta(perturb_str)
                        perturbed = gen_prompt_by_perturb_meta(doc, raw_tags, \
                                perturb_meta, meta, frameset_path=frameset_path, \
                                common_keywords_by_tag = common_keywords_by_tag, \
                                return_sequence=return_sequence, is_training=True)
                        
                        if not perturbed:
                            continue

                        neg_cases_by_curr_tag.append(
                            Munch(instance_idx=instance_idx,
                            prompt=perturbed.prompt,
                            answer=perturbed.answer,
                            perturb_str=perturb_str,
                            perturb_meta=perturb_meta,
                            meta = perturbed.meta,
                            reward=-1,
                            base_dict=base_dict
                        ))

                        cases_by_curr_doc[perturbed.prompt] = True
            except Exception as e:
                print(e)
                num_skipped += 1
                continue
            all_prompts += cases_by_curr_tag
            if use_unlikelihood:
                all_prompts += neg_cases_by_curr_tag
            if do_print:
                logger.info(f"instance: {instance_idx}")
                logger.info(f"num is_subset so far: {num_subset}/{num_total}")
                logger.info(f"num skipped so far: {num_skipped}/{num_total}")
                logger.info(f"Sentence: {doc}")
                logger.info(f"Tags: {raw_tags}")
                for pos in cases_by_curr_tag:
                    logger.info(" + " + "ORIGINAL")
                    logger.info(f"   \tis_subset: {pos.is_subset}")
                    logger.info(f"   \tnum_blank_args: {pos.num_blank_args}")
                    logger.info("   \t" + pos.prompt)
                    logger.info("   \t" + pos.answer)
                for neg in neg_cases_by_curr_tag:
                    logger.info(" - " + neg.perturb_str)
                    logger.info("   \t" + neg.prompt)
                    logger.info("   \t" + neg.answer)

    if is_return_pre_weight_prompts:
        return all_prompts
    # get the weights
    logger.info("Computing weights...")
    # TODO: need to change compute_weights to be able to handle new prompt/meta format (computing class labels does not work)
    if use_resampling:
        raise NotImplementedError
        all_prompts = compute_weights(
            all_prompts, 
            is_resample_by_label=is_resample_by_label and use_resampling,
            is_resample_by_feature=is_resample_by_feature and use_resampling,
            is_ngrams_as_features=is_ngrams_as_features,
            min_ngram=min_ngram,
            max_ngram=max_ngram,
        )
    else:
        for p in all_prompts:
            p.weight = 1
    groups = itertools.groupby(sorted(all_prompts, key=lambda p: p.instance_idx), key=lambda p: p.instance_idx)
    #groups = itertools.groupby(all_prompts, key=lambda p: p.instance_idx)
    groups = list([list(g) for _, g in groups])
    # eventually return in grouped format so when do train/dev split, will also be in group
    return groups


def compute_weights(all_prompts, 
    is_resample_by_label=True, is_resample_by_feature=True, 
    is_ngrams_as_features=True,
    min_ngram=1, max_ngram=3):
    weights = np.ones(len(all_prompts))
    class_labels = np.array([get_class_by_meta(t.meta, t.reward) for t in all_prompts])
    # two level of resampling: first by labels
    if is_resample_by_label:
        label_length_dict = dict(Counter(class_labels))
        label_mix_weight_dict = temperature_scale_weight(label_length_dict)
    else:
        label_mix_weight_dict = {label: 1 for label in set(class_labels)}
    # then by examples
    feature_extract = None
    analyzer_func = functools.partial(
        feature_extract, is_ngrams_as_features=is_ngrams_as_features, 
        min_ngram=min_ngram, max_ngram=max_ngram)
    if is_resample_by_feature:
        for class_label in tqdm(label_mix_weight_dict):
            label_mix_weight = label_mix_weight_dict[class_label]
            # find cases where the class label is the given one
            inclass_instance_idxes = np.where(class_labels == class_label)[0]
            #if class_label.startswith("UL:"):
            #    # don't try to resample unlikelihood examples
            #    for instance_idx in inclass_instance_idxes:
            #        weights[instance_idx] = label_mix_weight / 2
            corpus = [all_prompts[i] for i in inclass_instance_idxes]
            try:
                # try to exlude top words
                vectorizer = CountVectorizer(analyzer=analyzer_func, max_df=0.9)
                X = vectorizer.fit_transform(corpus)
            except:
                try:
                    vectorizer = CountVectorizer(analyzer=analyzer_func)
                    X = vectorizer.fit_transform(corpus)
                except:
                    vectorizer = CountVectorizer(analyzer=functools.partial(analyzer_func, is_filter_stopwords=False))
                    X = vectorizer.fit_transform(corpus)
            # compute the count of the features
            features = vectorizer.get_feature_names()
            feature_counts = np.array(X.sum(axis=0))[0]
            # the features included in one specific case
            instance_word_freq_dict = {}
            for corpus_idx, instance_idx in enumerate(inclass_instance_idxes):
                #print(train_prompts[instance_idx].prompt)
                #print(train_prompts[instance_idx].answer)
                #print(X[corpus_idx, :])
                included_features = list(find(X[corpus_idx, :])[1])
                # get the max count to represent the goal
                if included_features:
                    max_freq = max([feature_counts[idx] for idx in included_features])
                else:
                    max_freq = max(feature_counts)
                instance_word_freq_dict[instance_idx] = max_freq
            
            instance_mix_weight_dict = temperature_scale_weight(instance_word_freq_dict)
            for instance_idx, instance_mix_weight in instance_mix_weight_dict.items():
                weights[instance_idx] = instance_mix_weight * label_mix_weight
                # de-weight the negative samples a bit more?
                if class_label.startswith("UL:"):
                    weights[instance_idx] /= 10
    # normalize weights
    #max_weight = sum(weights)
    #weights = [w/max_weight for w in weights]
    # save the weights
    for prompt, weight in zip(all_prompts, weights):
        prompt.weight = weight
    return all_prompts





def temperature_scale_weight(label_length_dict, temperature=2.0, maximum=None, scale=1.0):
    '''Calculate mixing ratesm, maximum is k in the paper'''
    mixing_rates = {}
    for label, length in label_length_dict.items():
        rate = length * scale
        if maximum:
            rate = min(rate, maximum)
        if temperature != 1.0:
            rate = rate ** (1.0/temperature)
        mixing_rates[label] = rate
    # normalize and translate mixing ratio to weight
    min_val = min(list(mixing_rates.values()))
    for k in mixing_rates:
        mixing_rates[k] = min_val / mixing_rates[k]
    return mixing_rates

def get_class_by_meta(meta, reward):
    neg = "UL:" if reward == -1 else ""
    if meta.is_core: return f"{neg}VERB:{meta.vform}"
    else: return f"{neg}MODIFIER:{meta.tag}"


import dill as pickle

def write_prompt_files(prompt_groups, dpath, is_pickle=True):
    trains, devs = train_test_split(prompt_groups, test_size=0.1, random_state=42)
    train_list = list(np.concatenate(trains))
    dev_list = list(np.concatenate(devs))
    logger.info("Is writing csv training and dev files...")
    columns = ["prompt", "answer", "reward", "weight"]
    train_file = os.path.join(dpath, "train.csv")
    dev_file = os.path.join(dpath, "train.csv")
    pd.DataFrame(train_list)[columns].to_csv(train_file, index=False)
    pd.DataFrame(dev_list)[columns].to_csv(dev_file, index=False)
    logger.info(f" | Wrote train to: {train_file}")
    logger.info(f" | Wrote dev to: {dev_file}")
    logger.info("Is pickling the results...")
    if is_pickle:
        with open(f"{dpath}/train.pkl", "wb") as f:
            pickle.dump(trains, f)
        with open(f"{dpath}/dev.pkl", "wb") as f:
            pickle.dump(devs, f)
    logger.info(f"# train: {len(trains)}")
    logger.info(f"# dev: {len(devs)}")

def get_input_prompts_from_dev(dev_path, get_prompts_func):
    # get prompts from dev path
    df = pd.read_csv(dev_path)
    prompt_dicts = []
    for pid, prompt in tqdm(enumerate(df["prompt"].tolist())):
        prompts = get_prompts_func(prompt)
        for p in prompts:
            d = {"pid": pid}
            d.update(p)
            prompt_dicts.append(d)
    return prompt_dicts

def get_args():
    """Get the user arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', required=True, 
            choices=['preprocess', 'input'],
            help='The process to run. [preprocess] data, or create [input]')
    parser.add_argument('--data_path',  default="../../data")
    parser.add_argument('--prompt_identifier', help="The prompt identifier. Name of data subfolder.")
    parser.add_argument('--prompt_subdir', default="tailor/", type=str)
    parser.add_argument('--use_unlikelihood', default=False, action='store_true')
    parser.add_argument('--use_resampling', default=False, action='store_true')
    parser.add_argument('--frameset_path', help="The path to the frameset.", default=DEFAULT_FRAME_SET_PATH)
    parser.add_argument('--n_prompts', help="Number of total prompts to process", type=int, default=None)

    args = parser.parse_args()
    if args.step == 'input':
        if args.prompt_identifier is None:
            raise ValueError('Need to give identifier for prompts')
    return args

def write_args(args, dpath):
    json.dump(vars(args), open(os.path.join(dpath, "args.json"), "w"), indent=4, sort_keys=True)
    logger.info("Writing args to file:")
    logger.info(f" | {dpath}")

if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(
        format="%(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ]
    )

    if args.step == "preprocess":
        ONTONOTES_PATH = os.path.join(args.data_path, "conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0")
        SRLPATH = os.path.join(args.data_path, "orig")
        if not os.path.exists(SRLPATH):
            os.makedirs(SRLPATH)
        logger.info("Converting CONLL to json. This step takes some time. Format in readme.")
        logger.info("Processing train data...")
        convert_all_conll_files(
            f"{ONTONOTES_PATH}/data/train/data/english/annotations/", 
            f"{SRLPATH}/train.json")

        logger.info("Processing dev data...")
        convert_all_conll_files(
            f"{ONTONOTES_PATH}/data/development/data/english/annotations/", 
            f"{SRLPATH}/dev.json")
        
        logger.info("Processing test data...")
        convert_all_conll_files(
            f"{ONTONOTES_PATH}/data/conll-2012-test/data/english/annotations/", 
            f"{SRLPATH}/test.json")
    else:
        if not os.path.exists(args.prompt_subdir):
            os.makedirs(args.prompt_subdir, exist_ok=False)
        with open(os.path.join(args.data_path, "orig", "train.json"), "r") as f:
            data = json.load(f)
        # processor
        nlp = get_spacy_model("en_core_web_sm")
        logger.info(f"use unlikelihood: {args.use_unlikelihood}")
        logger.info(f"use resampling: {args.use_resampling}")
        logger.info(f"frameset path: {args.frameset_path}")
        prompt_groups = get_head_prompts_from_json(
            data[0:10000], nlp,return_sequence=True, 
            use_resampling=args.use_resampling,
            use_unlikelihood=args.use_unlikelihood, 
            frameset_path=args.frameset_path, n=args.n_prompts)
        dpath = os.path.join(args.data_path, args.prompt_subdir, args.prompt_identifier)
        if not os.path.exists(dpath):
            os.makedirs(dpath)
        write_args(args, dpath)
        write_prompt_files(prompt_groups, dpath)
