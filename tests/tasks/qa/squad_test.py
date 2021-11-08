import spacy
from tango.common.dataset_dict import DatasetDict

from tailor.common.testing import TailorTestCase
from tailor.tasks.qa.squad import LoadSquad, ProcessSquadWithSpacy


class TestSquad(TailorTestCase):
    def test_load_squad(self):
        step = LoadSquad(cache_results=False)
        result = step.run(data_dir=str(self.FIXTURES_ROOT / "data"), splits=["squad_dev_small"])

        assert isinstance(result, DatasetDict)
        assert len(result.splits) == 1
        assert "squad_dev_small" in result.splits
        assert len(result.splits["squad_dev_small"]) == 20

    def test_process_squad(self):
        step1 = LoadSquad(cache_results=False)
        squad = step1.run(data_dir=str(self.FIXTURES_ROOT / "data"), splits=["squad_dev_small"])

        step2 = ProcessSquadWithSpacy()
        result = step2.run(
            dataset_dict=squad, spacy_model_name="en_core_web_sm", keys_to_process=["answer"]
        )

        assert isinstance(result.splits["squad_dev_small"][0]["paragraph"], str)
        assert isinstance(result.splits["squad_dev_small"][0]["answer"], spacy.tokens.doc.Doc)
