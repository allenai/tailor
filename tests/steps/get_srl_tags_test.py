from tailor.common.testing import TailorTestCase
from tailor.common.utils import get_spacy_model
from tailor.steps.get_srl_tags import GetSRLTags, ProcessedSentence


class TestGetSRLTags(TailorTestCase):
    def test_step(self):

        nlp = get_spacy_model("en_core_web_sm")
        spacy_outs = [nlp(sent) for sent in ["Hi this is a test.", "Sample input text"]]

        step = GetSRLTags()
        tagged = step.run(spacy_outputs=spacy_outs)

        # assert tagged == [[['B-ARGM-DIS', 'B-ARG1', 'B-V', 'B-ARG2', 'I-ARG2', 'O']], []]
        assert len(tagged) == 2

        assert isinstance(tagged[0], ProcessedSentence)
        assert tagged[0].get_tags_list() == [
            ["B-ARGM-DIS", "B-ARG1", "B-V", "B-ARG2", "I-ARG2", "O"]
        ]
