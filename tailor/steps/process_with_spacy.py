from typing import Iterable, List

from tango.step import Step

from tailor.common.utils import get_spacy_model, SpacyDoc, SpacyModelType


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
    This step simply returns a spacy model, and optionally replaces its tokenizer
    with a white-space tokenizer.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "get-spacy-model".
    """

    DETERMINISTIC = True
    CACHEABLE = False  # Caching is unnecessary.

    def run(
        self,
        spacy_model_name: str = "en_core_web_sm",
        parse: bool = False,
        use_white_space_tokenizer: bool = False,
        **spacy_kwargs,
    ) -> SpacyModelType:
        """
        Returns a (possibly cached) spacy model.

        Parameters
        ----------

        spacy_model_name : :class:`str`
            The name of the spacy model. Default is `"en_core_web_sm"`.
        parse : :class:`bool`
            Whether the model does dependency parsing. Default is `False`.
            Set this to `True` if your tailor perturbation requires noun chunks.
        use_white_space_tokenizer : :class:`bool`
            Whether to use a white space tokenizer instead of spacy's tokenizer.
            Default is `False`.
            Set this to `True` when you have gold data which is pre-tokenized,
            but spacy's tokenization doesn't match the gold.

        Returns
        -------
        :class:`SpacyModelType`
            The spacy model.

        """
        spacy_model = get_spacy_model(
            spacy_model_name,
            parse=parse,
            updated_tokenizer=use_white_space_tokenizer,
            **spacy_kwargs,
        )
        if use_white_space_tokenizer:
            spacy_model.tokenizer = _WhitespaceSpacyTokenizer(spacy_model.vocab)
        return spacy_model


@Step.register("process-with-spacy")
class ProcessWithSpacy(Step):
    """
    This step applies the spacy model to the provided list of sentences.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "process-with-spacy".
    """

    DETERMINISTIC = True
    CACHEABLE = True
    VERSION = "001"

    def run(
        self,
        sentences: Iterable[str],
        spacy_model: SpacyModelType,
    ) -> List[SpacyDoc]:
        """
        Returns the list of spacy docs for all `sentences`.

        Parameters
        ----------
        sentences : :class:`Iterable[str]`,
            The list of sentences to process.
        spacy_model : :class:`SpacyModelType`
            The spacy model to use for processing the strings.

        Returns
        -------
        :class: `List[SpacyDoc]`
            The spacy docs for all strings.
        """

        outputs: List[SpacyDoc] = []
        for string in sentences:
            spacy_doc = spacy_model(string)
            outputs.append(spacy_doc)

        return outputs
