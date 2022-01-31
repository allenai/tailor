import torch
from tango.step import Step
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from transformers.pipelines import Text2TextGenerationPipeline
from tailor.common.model_utils import load_generator


@Step.register("load-generator")
class LoadTailorGenerator(Step):
    DETERMINISTIC = True
    CACHEABLE = (
        False  # We are loading a pretrained model using HF pipeline, no need to cache again.
    )

    def run(self, model_path: str = "allenai/tailor") -> Text2TextGenerationPipeline:
        return load_generator(model_path)
