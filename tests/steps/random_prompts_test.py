from tailor.common.abstractions import ProcessedSentence, PromptObject
from tailor.common.testing import TailorTestCase
from tailor.common.utils import get_spacy_model
from tailor.steps.random_prompts import GenerateRandomPrompts, GetCommonKeywordsByTag


class TestRandomPrompts(TailorTestCase):
    def setup_method(self):
        super().setup_method()

        nlp = get_spacy_model("en_core_web_sm", parse=True)
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

    def test_step(self):
        step = GetCommonKeywordsByTag()
        common_keywords = step.run()

        step = GenerateRandomPrompts()
        result = step.run(
            processed_sentences=self.processed_sentences, common_keywords_by_tag=common_keywords
        )

        assert len(result) == 2
        assert isinstance(result[0][0], PromptObject)
