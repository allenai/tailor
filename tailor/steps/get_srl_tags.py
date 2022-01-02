# import glob
# import json
# import os
import re
from typing import Dict, Iterable, List, Optional

import torch

# from tango.common import DatasetDict
from allennlp.predictors import Predictor
from tango.step import Step

from tailor.common.util import get_srl_tagger


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
        sentences: Iterable[str],
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

        input_jsons = [{"sentence": sentence} for sentence in sentences]
        with torch.no_grad():
            preds = []
            for idx in range(0, len(input_jsons), batch_size):
                preds += srl_tagger.predict_batch_json(input_jsons[idx : idx + batch_size])
            return preds


@Step.register("get-srl-tags")
class GetSRLTags(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        srl_predictions: Iterable[Dict],
    ) -> Iterable[List[List[str]]]:

        """
        Returns clean, unique tags for each verb in each srl prediction.
        """
        # TODO: do we always want unique tags? If not, then this should be a separate step.
        tags = []
        for pred in srl_predictions:
            tags.append([get_unique_tags(verb_dict["tags"]) for verb_dict in pred["verbs"]])
        return tags
