from munch import Munch
from tailor.common.testing import TailorTestCase

from tailor.common.abstractions import PromptObject, ProcessedSentence, GeneratedPrompt
from tailor.common.util import get_spacy_model
from tailor.steps.generate_from_prompts import GenerateFromPrompts


class TestGenerateFromPrompts(TailorTestCase):
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

        self.prompts = [
            [
                PromptObject(
                    prompt=(
                        "[VERB+passive+past: comfort | AGENT+complete: by the doctor | "
                        "PATIENT+complete: the patient]  "
                        "<extra_id_0> <extra_id_1> <extra_id_2> <extra_id_3> <extra_id_4> <extra_id_5> ."
                    ),
                    answer="[AGENT: The doctor] [VERB: comforted] [PATIENT: the patient] .",
                    meta=Munch(
                        {
                            "noncore_args": [],
                            "core_args": [
                                Munch(
                                    {
                                        "tlemma": "by the doctor",
                                        "tlemma_type": "complete",
                                        "raw_tag": "ARG0",
                                        "tag": "AGENT",
                                        "blank_idx": [0, 2],
                                    }
                                ),
                                Munch(
                                    {
                                        "tlemma": "the patient",
                                        "tlemma_type": "complete",
                                        "raw_tag": "ARG1",
                                        "tag": "PATIENT",
                                        "blank_idx": [3, 5],
                                    }
                                ),
                            ],
                            "blank_indexes": [[0, 2], [2, 3], [3, 5], [2, 2], [2, 2], [2, 2]],
                            "blank_appearance_indexes": [0, 2, 3, 2, 2, 2],
                            "answers": [
                                "[AGENT: The doctor]",
                                "[VERB: comforted]",
                                "[PATIENT: the patient]",
                                "",
                                "",
                                "",
                            ],
                            "vvoice": "passive",
                            "vlemma": "comfort",
                            "vtense": "past",
                            "doc": nlp("The doctor comforted the patient ."),
                            "raw_tags": ["B-ARG0", "I-ARG0", "B-V", "B-ARG1", "I-ARG1", "O"],
                            "frameset_id": None,
                        }
                    ),
                    name="change_voice",
                )
            ],
            [
                PromptObject(
                    prompt="[VERB+passive+past: be]  <extra_id_0> .",
                    answer="The book [VERB: was] picked by the girl .",
                    meta=Munch(
                        {
                            "noncore_args": [],
                            "core_args": [],
                            "blank_indexes": [[2, 3]],
                            "blank_appearance_indexes": [2],
                            "answers": ["[VERB: was]"],
                            "vvoice": "passive",
                            "vlemma": "be",
                            "vtense": "past",
                            "doc": nlp("The book was picked by the girl ."),
                            "raw_tags": ["O", "O", "B-V", "O", "O", "O", "O", "O"],
                            "frameset_id": None,
                        }
                    ),
                    name="change_voice",
                ),
                PromptObject(
                    prompt=(
                        "[VERB+passive+past: pick | AGENT+complete: by the girl | "
                        "PATIENT+complete: the book]  <extra_id_0> <extra_id_1> "
                        "<extra_id_2> <extra_id_3> <extra_id_4> <extra_id_5> <extra_id_6> "
                        "<extra_id_7> <extra_id_8> ."
                    ),
                    answer="[PATIENT: The book] was [VERB: picked] [AGENT: by the girl] .",
                    meta=Munch(
                        {
                            "noncore_args": [],
                            "core_args": [
                                Munch(
                                    {
                                        "tlemma": "by the girl",
                                        "tlemma_type": "complete",
                                        "raw_tag": "ARG0",
                                        "tag": "AGENT",
                                        "blank_idx": [4, 7],
                                    }
                                ),
                                Munch(
                                    {
                                        "tlemma": "the book",
                                        "tlemma_type": "complete",
                                        "raw_tag": "ARG1",
                                        "tag": "PATIENT",
                                        "blank_idx": [0, 2],
                                    }
                                ),
                            ],
                            "blank_indexes": [
                                [0, 2],
                                [4, 7],
                                [3, 4],
                                [3, 3],
                                [3, 3],
                                [3, 3],
                                [3, 3],
                                [3, 3],
                                [3, 3],
                            ],
                            "blank_appearance_indexes": [0, 4, 3, 3, 3, 3, 3, 3, 3],
                            "answers": [
                                "[PATIENT: The book]",
                                "[AGENT: by the girl]",
                                "[VERB: picked]",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                            ],
                            "vvoice": "passive",
                            "vlemma": "pick",
                            "vtense": "past",
                            "doc": nlp("The book was picked by the girl ."),
                            "raw_tags": [
                                "B-ARG1",
                                "I-ARG1",
                                "O",
                                "B-V",
                                "B-ARG0",
                                "I-ARG0",
                                "I-ARG0",
                                "O",
                            ],
                            "frameset_id": None,
                        }
                    ),
                    name="change_voice",
                ),
            ],
        ]

        self.nlp = nlp

    def test_step(self):

        step = GenerateFromPrompts()
        result = step.run(
            processed_sentences=self.processed_sentences,
            prompts=self.prompts,
            spacy_model=self.nlp,
        )

        assert len(result) == 2
        assert isinstance(result[0][0], GeneratedPrompt)
