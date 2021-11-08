import torch
from tango.step import Step
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from transformers.pipelines import Text2TextGenerationPipeline


@Step.register("load-generator")
class LoadTailorGenerator(Step):
    DETERMINISTIC = True
    CACHEABLE = (
        False  # We are loading a pretrained model using HF pipeline, no need to cache again.
    )

    def run(self, model_path: str = "allenai/tailor") -> Text2TextGenerationPipeline:

        pipename = "text2text-generation"
        tokenizer = AutoTokenizer.from_pretrained(model_path)  # TODO: should this be "t5-base"?
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        if torch.cuda.is_available():
            device = 0
        else:
            device = -1

        return pipeline(pipename, model=model, tokenizer=tokenizer, framework="pt", device=device)
