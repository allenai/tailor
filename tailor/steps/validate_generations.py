from typing import List, Optional, Tuple
import torch
from tango.step import Step


from tailor.common.abstractions import GeneratedPrompt

# from tailor.common.utils import SpacyModelType  # , get_srl_tagger, predict_batch_srl
from tailor.common.filters.perplex_filter import load_perplex_scorer

# from tailor.common.filters.ctrl_filter import is_followed_ctrl
# from tailor.common.utils.head_prompt_utils import add_predictions_to_prompt_dict_new

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase


@Step.register("load-perplexity-scorer")
class LoadPerplexityScorer(Step):
    DETERMINISTIC = True
    CACHEABLE = False  # TODO

    def run(self, model_name: str = "gpt2") -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        perplex_scorer = load_perplex_scorer(is_cuda=torch.cuda.is_available())
        return perplex_scorer.model, perplex_scorer.tokenizer


@Step.register("validate-generations")
class ValidateGenerations(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    VERSION = "05"

    def run(
        self,
        generated_prompt_dicts: List[List[GeneratedPrompt]],
        perplex_thresh: Optional[int] = None,
    ) -> List[List[str]]:

        # srl_tagger = get_srl_tagger()  # TODO

        all_sentences = []

        for idx, sentence_prompts in enumerate(generated_prompt_dicts):
            validated_set = []
            for prompt_dict in sentence_prompts:
                is_valid = True
                perplexity = prompt_dict.perplexities
                if perplex_thresh is not None and perplexity is not None:
                    is_valid = (
                        perplexity.pr_sent < perplex_thresh
                        and perplexity.pr_phrase < perplex_thresh
                    )
                # TODO (Alexis): Is this required?
                # if is_valid:
                #     predicted = predict_batch_srl([generated.strip()], srl_tagger)[0]
                #     prompt_dict = add_predictions_to_prompt_dict_new(prompt_dict, predicted)
                #     is_valid = is_valid and is_followed_ctrl(prompt_dict, generated_doc, spacy_model)

                if is_valid:
                    validated_set.append(prompt_dict.sentence)

            all_sentences.append(validated_set)
        return all_sentences
