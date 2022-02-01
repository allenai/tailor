from tailor.common.testing import TailorTestCase
from tailor.common.utils import get_spacy_model
from tailor.common.utils.detect_perturbations import get_common_keywords_by_tag
from tailor.common.abstractions import ProcessedSentence

from tailor.steps.random_prompts import GenerateRandomPrompts

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
        self.common_keywords = get_common_keywords_by_tag(nlp=nlp)

    def test_step(self):
        step = GenerateRandomPrompts()
        result = step.run(processed_sentences=self.processed_sentences, common_keywords_by_tag=self.common_keywords)

        print(result)
        assert False
