from tailor.common.testing import TailorTestCase
from tailor.common.utils import get_spacy_model
from tailor.common.utils.head_prompt_utils import gen_prompt_by_perturb_str, is_equal_headers
from tailor.common.abstractions import ProcessedSentence
from tailor.common.perturb_function import ChangeVoice, PerturbFunction
from tailor.steps.perturb_prompt import (
    PerturbPromptWithString,
    PerturbPromptWithFunction,
    PromptObject,
    _munch_to_prompt_object,
)


class SamplePerturbFunction(PerturbFunction):
    def __call__(self, spacy_doc, intermediate_prompt, tags, *args, **kwargs):
        vtense = intermediate_prompt.meta.vtense
        target_voice = "active" if intermediate_prompt.meta.vvoice == "passive" else "passive"
        perturb_str = (
            f"CONTEXT(DELETE_TEXT);VERB(CHANGE_TENSE({vtense}),CHANGE_VOICE({target_voice}))"
        )

        perturbed = gen_prompt_by_perturb_str(
            spacy_doc, tags, perturb_str, intermediate_prompt.meta
        )

        if perturbed is None:
            return None
        if not is_equal_headers(perturbed.prompt, intermediate_prompt.prompt):
            return _munch_to_prompt_object(perturbed, name="sample")


class TestPerturbPrompt(TailorTestCase):
    def setup_method(self):
        super().setup_method()

        nlp = get_spacy_model("en_core_web_sm")
        sentences = ["The doctor comforted the patient .", "The book was picked by the girl ."]
        verbs = [
            [
                {
                    "verb": "comforted",
                    "description": "[ARG0: The doctor] [V: comforted] [ARG1: the patient] .",
                    "tags": ["B-ARG0", "I-ARG0", "B-V", "B-ARG1", "I-ARG1", "O"],
                }
            ],
            [
                {
                    "verb": "was",
                    "description": "The book [V: was] picked by the girl .",
                    "tags": ["O", "O", "B-V", "O", "O", "O", "O", "O"],
                },
                {
                    "verb": "picked",
                    "description": "[ARG1: The book] was [V: picked] [ARG0: by the girl] .",
                    "tags": ["B-ARG1", "I-ARG1", "O", "B-V", "B-ARG0", "I-ARG0", "I-ARG0", "O"],
                },
            ],
        ]

        processed = []
        for idx, sentence in enumerate(sentences):
            processed.append(
                ProcessedSentence(sentence=sentence, spacy_doc=nlp(sentence), verbs=verbs[idx])
            )

        self.processed_sentences = processed

    def test_with_string(self):

        step = PerturbPromptWithString()
        result = step.run(
            processed_sentences=self.processed_sentences,
            perturb_str_func="CONTEXT(DELETE_TEXT);VERB(CHANGE_VOICE(passive))",
        )

        assert len(result) == 2
        assert isinstance(result[0][0], PromptObject)

    def test_with_string_function(self):

        step = PerturbPromptWithString()
        result = step.run(
            processed_sentences=self.processed_sentences,
            perturb_str_func=ChangeVoice(),
        )

        assert len(result) == 2
        assert isinstance(result[0][0], PromptObject)

    def test_with_function(self):
        step = PerturbPromptWithFunction()
        result = step.run(
            processed_sentences=self.processed_sentences,
            perturb_fn=SamplePerturbFunction(),
        )

        assert len(result) == 2
        assert isinstance(result[0][0], PromptObject)
