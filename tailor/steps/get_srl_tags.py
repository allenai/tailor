# import glob
# import json
# import os
import re
from typing import Dict, Iterable, List, NamedTuple, Optional

import torch

from tango.common import DatasetDict
from allennlp.predictors import Predictor
from tango.step import Step

from tailor.common.util import get_srl_tagger
from tailor.steps.process_with_spacy import SpacyOutput, SpacyDoc


def clean_prefix_for_one_tag(tag: str) -> str:
    """delete I/B- prefix for the BIO tag

    Args:
        tag (str): SRL tag

    Returns:
        str: the SRL tag with I/B- removed
    """
    return re.split(r"(B|I)\-", tag)[-1]


def get_unique_tags(raw_tags: List[str]) -> List[str]:
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


@Step.register("get-srl-predictions")
class GetSRLPredictions(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        spacy_outputs: Iterable[SpacyOutput],
        srl_tagger: Optional[Predictor] = None,
        batch_size: int = 128,
    ) -> Iterable[Dict]:
        """
        Returns output in the following format
        ```
        [
            {"words": [...],
             "verbs": [
                {"verb": "...", "description": "...", "tags": [...]},
                ...
                {"verb": "...", "description": "...", "tags": [...]},
            ]},
            {"words": [...],
             "verbs": [
                {"verb": "...", "description": "...", "tags": [...]},
                ...
                {"verb": "...", "description": "...", "tags": [...]},
            ]}
        ]
        ```
        """
        srl_tagger = srl_tagger or get_srl_tagger()

        input_jsons = [
            {"sentence": spacy_output.updated_sentence} for spacy_output in spacy_outputs
        ]
        with torch.no_grad():
            preds = []
            for idx in range(0, len(input_jsons), batch_size):
                preds += srl_tagger.predict_batch_json(input_jsons[idx : idx + batch_size])
            return preds


# TODO: instead of Iterable list/dict, etc. create a specific format.
# @Step.register("get-srl-tags")
# class GetSRLTags(Step):
#     DETERMINISTIC = True
#     CACHEABLE = True

#     def run(
#         self,
#         srl_predictions: Iterable[Dict],
#     ) -> Iterable[List[List[str]]]:

#         """
#         Returns clean, unique tags for each verb in each srl prediction.
#         """
#         # TODO: do we always want unique tags? If not, then this should be a separate step.
#         tags = []
#         for pred in srl_predictions:
#             tags.append([get_unique_tags(verb_dict["tags"]) for verb_dict in pred["verbs"]])
#         return tags


class PerturbationError(Exception):
    pass


class ProcessedSentence(NamedTuple):

    sentence: str
    spacy_doc: SpacyDoc
    verbs: List[Dict]  # Dict: {"verb": str, "tags": List[str]}

    def get_tags_list(self):
        return [verb_dict["tags"] for verb_dict in self.verbs]

    def extract_relative_clauses_tags(self):
        """Extracts all verbs with an argument that includes a relative clause
        Args:
            pred: output of calling predict function on AllenNLP SRL predictor
        Returns:
            list[list[str]]: list of list of tags
                Each sub-list corresponds to a different verb
        """
        rel_idxes = [
            idx
            for idx, verb_dict in enumerate(self.verbs)
            if any(t.startswith("B-R") for t in verb_dict["tags"])
        ]
        if not rel_idxes:
            raise PerturbationError
        tags_lst = [self.verbs[rel_idx]["tags"] for rel_idx in rel_idxes]
        return tags_lst


@Step.register("srl-tags")
class SRLTags(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        spacy_outputs: Iterable[SpacyOutput],
        srl_tagger: Optional[Predictor] = None,
    ) -> Iterable[List[Dict]]:  # Multiple verbs for each sentence.
        srl_tagger = srl_tagger or get_srl_tagger()
        outputs = []
        # with torch.no_grad():
        for spacy_output in spacy_outputs:
            sentence = spacy_output.updated_sentence
            srl_prediction = srl_tagger.predict_json(
                {"sentence": sentence}
            )  # TODO: use batches for efficiency.

            outputs.append(
                ProcessedSentence(sentence, spacy_output.spacy_doc, srl_prediction["verbs"])
            )
            # verb_tags = []
            # for verb_dict in srl_prediction["verbs"]:
            #     srl_tags = verb_dict["tags"]
            #     verb_tags.append(srl_tags)

            # outputs.append(verb_tags)
        return outputs
