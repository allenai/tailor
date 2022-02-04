from transformers.pipelines import Text2TextGenerationPipeline

from tailor.common.testing import TailorTestCase
from tailor.steps import LoadTailorGenerator


class TestLoadGenerator(TailorTestCase):
    def test_load_generator(self):
        step = LoadTailorGenerator()
        result = step.run()

        assert isinstance(result, Text2TextGenerationPipeline)
        prompt_text = "[VERB+active+past: comfort | "
        prompt_text += "AGENT+complete: the doctor | "
        prompt_text += "PATIENT+partial: athlete | "
        prompt_text += (
            "LOCATIVE+partial: in] <extra_id_0> , <extra_id_1> <extra_id_2> <extra_id_3> ."
        )
        output = result(prompt_text, max_length=200)
        assert "generated_text" in output[0]
