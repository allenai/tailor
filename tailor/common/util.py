import logging
from typing import Dict, Tuple

from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.predictors import Predictor
import cached_path

import spacy
from spacy.cli.download import download as spacy_download
from spacy.language import Language as SpacyModelType
from spacy.tokens.doc import Doc as SpacyDoc  # noqa: F401

import torch

logger = logging.getLogger(__name__)

LOADED_SPACY_MODELS: Dict[Tuple[str, bool, bool, bool], SpacyModelType] = {}


def get_spacy_model(
    spacy_model_name: str, pos_tags: bool = True, parse: bool = False, ner: bool = False
) -> SpacyModelType:
    """
    Copied over from allennlp.common.util

    In order to avoid loading spacy models a whole bunch of times, we'll save references to them,
    keyed by the options we used to create the spacy model, so any particular configuration only
    gets loaded once.
    """

    options = (spacy_model_name, pos_tags, parse, ner)
    if options not in LOADED_SPACY_MODELS:
        disable = ["vectors", "textcat"]
        if not pos_tags:
            disable.append("tagger")
        if not parse:
            disable.append("parser")
        if not ner:
            disable.append("ner")
        try:
            spacy_model = spacy.load(spacy_model_name, disable=disable)
        except OSError:
            logger.warning(
                f"Spacy models '{spacy_model_name}' not found.  Downloading and installing."
            )
            spacy_download(spacy_model_name)

            # Import the downloaded model module directly and load from there
            spacy_model_module = __import__(spacy_model_name)
            spacy_model = spacy_model_module.load(disable=disable)  # type: ignore

        LOADED_SPACY_MODELS[options] = spacy_model
    return LOADED_SPACY_MODELS[options]


def get_srl_tagger(
    model_path: str = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
):
    """
    Returns an AllenNLP predictor for getting SRL tags.
    """
    if torch.cuda.is_available():
        cuda_device = 0
    else:
        cuda_device = -1

    predictor = Predictor.from_path(
        cached_path.cached_path(model_path), cuda_device=cuda_device, frozen=True
    )
    predictor._tokenizer = SpacyTokenizer(pos_tags=True, split_on_spaces=True)
    return predictor
