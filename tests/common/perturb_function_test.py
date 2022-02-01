import copy
from munch import Munch
from tailor.common.utils import get_spacy_model
from tailor.common.testing import TailorTestCase
from tailor.common.perturbation_criteria import UniqueTags
from tailor.common.perturb_function import *


class TestPerturbFunction(TailorTestCase):
    def setup_method(self):
        super().setup_method()
        nlp = get_spacy_model("en_core_web_sm", parse=True)

        # sentences = ["The doctor comforted the patient .", "The book was picked by the girl ."]

        self.verbs = [
            {
                "verb": "comforted",
                "description": "[ARG0: The doctor] [V: comforted] [ARG1: the patient] .",
                "tags": ["B-ARG0", "I-ARG0", "B-V", "B-ARG1", "I-ARG1", "O"],
            }
        ]

        self.prompt_meta = Munch(
            {
                "noncore_args": [],
                "core_args": [
                    Munch(
                        {
                            "tlemma": "the doctor",
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
                "blank_indexes": [[2, 3], [3, 5], [0, 2]],
                "blank_appearance_indexes": [2, 3, 0],
                "answers": ["[VERB: comforted]", "[PATIENT: the patient]", "[AGENT: The doctor]"],
                "vvoice": "active",
                "vlemma": "comfort",
                "vtense": "past",
                "doc": nlp("The doctor comforted the patient ."),
                "raw_tags": ["B-ARG0", "I-ARG0", "B-V", "B-ARG1", "I-ARG1", "O"],
                "frameset_id": None,
            }
        )

    def test_change_voice(self):
        meta = copy.deepcopy(self.prompt_meta)
        perturb_fn = ChangeVoice()
        perturb_str = perturb_fn(meta).perturb_str

        assert perturb_str == "CONTEXT(DELETE_TEXT);VERB(CHANGE_TENSE(past),CHANGE_VOICE(passive))"

    def test_change_tense(self):
        meta = copy.deepcopy(self.prompt_meta)
        perturb_fn = ChangeTense()
        perturb_str = perturb_fn(meta).perturb_str

        assert perturb_str == "VERB(CHANGE_TENSE())"

    def test_shorten_core_argument(self):
        meta = copy.deepcopy(self.prompt_meta)
        perturb_fn = ShortenCoreArgument()

        perturbs = perturb_fn(meta)

        assert len(perturbs) == 2  # agent/patient.
        assert (
            perturbs[0].perturb_str
            == "CONTEXT(DELETE_TEXT);NONCORE(ALL:DELETE);CORE(AGENT:CHANGE_CONTENT(The doctor),CHANGE_SPECIFICITY(complete))"
        )
