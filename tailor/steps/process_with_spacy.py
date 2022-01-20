from typing import Iterable

from tango.step import Step

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
    """
    TODO: Docs
    """

    DETERMINISTIC = True
    CACHEABLE = False  # Caching is unnecessary.

    def run(
        self,
        spacy_model_name: str = "en_core_web_sm",
        use_white_space_tokenizer: bool = False,
        **spacy_kwargs,
    ) -> SpacyModelType:
        spacy_model = get_spacy_model(spacy_model_name, **spacy_kwargs)
        if use_white_space_tokenizer:
            spacy_model.tokenizer = _WhitespaceSpacyTokenizer(spacy_model.vocab)
        return spacy_model


@Step.register("process-with-spacy")
class ProcessWithSpacy(Step):
    """
    TODO: Docs
    """

    DETERMINISTIC = True
    CACHEABLE = True
    VERSION = "001"

    def run(
        self,
        sentences: Iterable[str],
        spacy_model: SpacyModelType,
    ) -> Iterable[SpacyDoc]:

        outputs: Iterable[SpacyDoc] = []
        for string in sentences:
            spacy_doc = spacy_model(string)
            outputs.append(spacy_doc)

        return outputs
