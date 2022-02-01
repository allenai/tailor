from munch import Munch
from tailor.common.testing import TailorTestCase

from tailor.common.abstractions import ProcessedSentence, GeneratedPrompt
from tailor.common.util import get_spacy_model
from tailor.steps.validate_generations import ValidateGenerations


class TestValidateGenerations(TailorTestCase):
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

        self.nlp = nlp

        self.generated_prompts = [
            [
                GeneratedPrompt(
                    prompt_no_header="- [PATIENT: the patient] is [VERB: comforted], [AGENT : by the doctor]",
                    sentence="- the patient is comforted , by the doctor",
                    meta=Munch(
                        {
                            "match": "<re.Match object; span=(0, 28), match='[VERB+passive+past: comfort '>",
                            "vlemma": "comfort",
                            "vvoice": "passive",
                            "vtense": "past",
                            "noncore_args": [],
                            "core_args": [
                                Munch(
                                    {
                                        "tlemma": "by the doctor",
                                        "tlemma_type": "complete",
                                        "raw_tag": None,
                                        "tag": "AGENT",
                                        "blank_idx": None,
                                    }
                                ),
                                Munch(
                                    {
                                        "tlemma": "the patient",
                                        "tlemma_type": "complete",
                                        "raw_tag": None,
                                        "tag": "PATIENT",
                                        "blank_idx": None,
                                    }
                                ),
                            ],
                        }
                    ),
                    annotations=[
                        Munch({"tag": "PATIENT", "start": 1, "end": 3, "pred": ""}),
                        Munch({"tag": "VERB", "start": 4, "end": 5, "pred": ""}),
                        Munch({"tag": "AGENT", "start": 6, "end": 9, "pred": ""}),
                    ],
                    words=["-", "the", "patient", "is", "comforted", ",", "by", "the", "doctor"],
                    vidx=4,
                    name=None,
                ),
                GeneratedPrompt(
                    prompt_no_header="Having's [VERB: comforted] - [PATIENT: the patient] [AGENT : by the doctor]",
                    sentence="Having 's comforted - the patient by the doctor",
                    meta=Munch(
                        {
                            "match": "<re.Match object; span=(0, 28), match='[VERB+passive+past: comfort '>",
                            "vlemma": "comfort",
                            "vvoice": "passive",
                            "vtense": "past",
                            "noncore_args": [],
                            "core_args": [
                                Munch(
                                    {
                                        "tlemma": "by the doctor",
                                        "tlemma_type": "complete",
                                        "raw_tag": None,
                                        "tag": "AGENT",
                                        "blank_idx": None,
                                    }
                                ),
                                Munch(
                                    {
                                        "tlemma": "the patient",
                                        "tlemma_type": "complete",
                                        "raw_tag": None,
                                        "tag": "PATIENT",
                                        "blank_idx": None,
                                    }
                                ),
                            ],
                        }
                    ),
                    annotations=[
                        Munch({"tag": "VERB", "start": 2, "end": 3, "pred": ""}),
                        Munch({"tag": "PATIENT", "start": 4, "end": 6, "pred": ""}),
                        Munch({"tag": "AGENT", "start": 6, "end": 9, "pred": ""}),
                    ],
                    words=[
                        "Having",
                        "'s",
                        "comforted",
                        "-",
                        "the",
                        "patient",
                        "by",
                        "the",
                        "doctor",
                    ],
                    vidx=2,
                    name=None,
                ),
                GeneratedPrompt(
                    prompt_no_header="- [PATIENT: the patient] was [VERB: comforted] [AGENT; by the doctor].",
                    sentence="- the patient was comforted by the doctor .",
                    meta=Munch(
                        {
                            "match": "<re.Match object; span=(0, 28), match='[VERB+passive+past: comfort '>",
                            "vlemma": "comfort",
                            "vvoice": "passive",
                            "vtense": "past",
                            "noncore_args": [],
                            "core_args": [
                                Munch(
                                    {
                                        "tlemma": "by the doctor",
                                        "tlemma_type": "complete",
                                        "raw_tag": None,
                                        "tag": "AGENT",
                                        "blank_idx": None,
                                    }
                                ),
                                Munch(
                                    {
                                        "tlemma": "the patient",
                                        "tlemma_type": "complete",
                                        "raw_tag": None,
                                        "tag": "PATIENT",
                                        "blank_idx": None,
                                    }
                                ),
                            ],
                        }
                    ),
                    annotations=[
                        Munch({"tag": "PATIENT", "start": 1, "end": 3, "pred": ""}),
                        Munch({"tag": "VERB", "start": 4, "end": 5, "pred": ""}),
                        Munch({"tag": "AGENT", "start": 5, "end": 8, "pred": ""}),
                    ],
                    words=["-", "the", "patient", "was", "comforted", "by", "the", "doctor", "."],
                    vidx=4,
                    name=None,
                ),
            ],
            [
                GeneratedPrompt(
                    prompt_no_header="- [PATIENT: the book]'s [VERB: picked] [AGENT : by the girl]",
                    sentence="- the book 's picked by the girl",
                    meta=Munch(
                        {
                            "match": "<re.Match object; span=(0, 25), match='[VERB+passive+past: pick '>",
                            "vlemma": "pick",
                            "vvoice": "passive",
                            "vtense": "past",
                            "noncore_args": [],
                            "core_args": [
                                Munch(
                                    {
                                        "tlemma": "by the girl",
                                        "tlemma_type": "complete",
                                        "raw_tag": None,
                                        "tag": "AGENT",
                                        "blank_idx": None,
                                    }
                                ),
                                Munch(
                                    {
                                        "tlemma": "the book",
                                        "tlemma_type": "complete",
                                        "raw_tag": None,
                                        "tag": "PATIENT",
                                        "blank_idx": None,
                                    }
                                ),
                            ],
                        }
                    ),
                    annotations=[
                        Munch({"tag": "PATIENT", "start": 1, "end": 3, "pred": ""}),
                        Munch({"tag": "VERB", "start": 4, "end": 5, "pred": ""}),
                        Munch({"tag": "AGENT", "start": 5, "end": 8, "pred": ""}),
                    ],
                    words=["-", "the", "book", "'s", "picked", "by", "the", "girl"],
                    vidx=4,
                    name=None,
                ),
                GeneratedPrompt(
                    prompt_no_header="- [PATIENT: the book]'s [VERB: picked] [AGENT- by the girl]",
                    sentence="- the book 's picked by the girl",
                    meta=Munch(
                        {
                            "match": "<re.Match object; span=(0, 25), match='[VERB+passive+past: pick '>",
                            "vlemma": "pick",
                            "vvoice": "passive",
                            "vtense": "past",
                            "noncore_args": [],
                            "core_args": [
                                Munch(
                                    {
                                        "tlemma": "by the girl",
                                        "tlemma_type": "complete",
                                        "raw_tag": None,
                                        "tag": "AGENT",
                                        "blank_idx": None,
                                    }
                                ),
                                Munch(
                                    {
                                        "tlemma": "the book",
                                        "tlemma_type": "complete",
                                        "raw_tag": None,
                                        "tag": "PATIENT",
                                        "blank_idx": None,
                                    }
                                ),
                            ],
                        }
                    ),
                    annotations=[
                        Munch({"tag": "PATIENT", "start": 1, "end": 3, "pred": ""}),
                        Munch({"tag": "VERB", "start": 4, "end": 5, "pred": ""}),
                        Munch({"tag": "AGENT", "start": 5, "end": 8, "pred": ""}),
                    ],
                    words=["-", "the", "book", "'s", "picked", "by", "the", "girl"],
                    vidx=4,
                    name=None,
                ),
                GeneratedPrompt(
                    prompt_no_header=". [PATIENT: the book]'s [VERB: picked] [AGENT : by the girl]",
                    sentence=". the book 's picked by the girl",
                    meta=Munch(
                        {
                            "match": "<re.Match object; span=(0, 25), match='[VERB+passive+past: pick '>",
                            "vlemma": "pick",
                            "vvoice": "passive",
                            "vtense": "past",
                            "noncore_args": [],
                            "core_args": [
                                Munch(
                                    {
                                        "tlemma": "by the girl",
                                        "tlemma_type": "complete",
                                        "raw_tag": None,
                                        "tag": "AGENT",
                                        "blank_idx": None,
                                    }
                                ),
                                Munch(
                                    {
                                        "tlemma": "the book",
                                        "tlemma_type": "complete",
                                        "raw_tag": None,
                                        "tag": "PATIENT",
                                        "blank_idx": None,
                                    }
                                ),
                            ],
                        }
                    ),
                    annotations=[
                        Munch({"tag": "PATIENT", "start": 1, "end": 3, "pred": ""}),
                        Munch({"tag": "VERB", "start": 4, "end": 5, "pred": ""}),
                        Munch({"tag": "AGENT", "start": 5, "end": 8, "pred": ""}),
                    ],
                    words=[".", "the", "book", "'s", "picked", "by", "the", "girl"],
                    vidx=4,
                    name=None,
                ),
            ],
        ]

    def test_step(self):
        step = ValidateGenerations()

        result = step.run(
            processed_sentences=self.processed_sentences,
            generated_prompt_dicts=self.generated_prompts,
            spacy_model=self.nlp,
        )

        assert len(result) == 2
        assert isinstance(result[0][0], str)
