import copy
from typing import Iterable, List, NamedTuple, Optional

from tango.step import Step
from tango.common import DatasetDict

# from tango.integrations.datasets import DatasetsFormat

from tailor.common.util import get_spacy_model, SpacyDoc, SpacyModelType


class _WhitespaceSpacyTokenizer:
    """
    Spacy doesn't assume that text is tokenized. Sometimes this
    is annoying, like when you have gold data which is pre-tokenized,
    but Spacy's tokenization doesn't match the gold. This can be used
    as follows:
    nlp = spacy.load("en_core_web_md")
    # hack to replace tokenizer with a whitespace tokenizer
    nlp.tokenizer = _WhitespaceSpacyTokenizer(nlp.vocab)
    ... use nlp("here is some text") as normal.
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.strip().split(" ")
        spaces = [True] * len(words)
        return SpacyDoc(self.vocab, words=words, spaces=spaces)


@Step.register("get-spacy-model")
class GetSpacyModel(Step):
    DETERMINISTIC = True
    CACHEABLE = True  # TODO: should it be?

    def run(
        self,
        spacy_model_name: str = "en_core_web_sm",
        use_white_space_tokenizer: bool = False,
    ) -> SpacyModelType:
        spacy_model = get_spacy_model(spacy_model_name)
        if use_white_space_tokenizer:
            spacy_model.tokenizer = _WhitespaceSpacyTokenizer(spacy_model.vocab)
        return spacy_model


@Step.register("process-dataset-with-spacy")
class ProcessDatasetWithSpacy(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    # FORMAT = DatasetsFormat()

    def run(
        self,
        spacy_model: SpacyModelType,
        dataset_dict: DatasetDict,
        key_to_process: str,
        processed_key_name: Optional[str] = None,
    ) -> DatasetDict:
        split_data: Dict[str, List] = {}

        processed_key_name = processed_key_name or key_to_process

        for split in dataset_dict:
            dataset = dataset_dict[split]

            split_data[split] = []

            for instance in dataset:
                processed_instance = copy.deepcopy(instance)
                processed_instance[processed_key_name] = spacy_model(
                    processed_instance[key_to_process]
                )
                split_data[split].append(processed_instance)

        return DatasetDict(split_data)


class SpacyOutput(NamedTuple):
    spacy_doc: SpacyDoc
    updated_sentence: Optional[str] = None


@Step.register("process-with-spacy")
class ProcessWithSpacy(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    VERSION = "001"

    def run(
        self,
        sentences: Iterable[str],
        spacy_model: SpacyModelType,
    ) -> Iterable[SpacyOutput]:

        outputs: Iterable[SpacyOutput] = []
        for string in sentences:
            spacy_doc = spacy_model(string)
            # if save_processed_text:
            updated_sentence = " ".join([token.text for token in spacy_doc])
            outputs.append(SpacyOutput(spacy_doc=spacy_doc, updated_sentence=updated_sentence))

        return outputs
