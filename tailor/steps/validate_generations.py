from munch import Munch
from typing import List, Optional, Tuple
import torch
from tango.step import Step

from allennlp.predictors import Predictor

from tailor.common.abstractions import ProcessedSentence, GeneratedPrompt

from tailor.common.util import SpacyModelType  # , get_srl_tagger, predict_batch_srl
from tailor.common.generate_utils import compute_edit_ops
from tailor.common.perplex_filter import (
    compute_delta_perplexity,
    # compute_sent_perplexity,
    load_perplex_scorer,
)

# from tailor.common.ctrl_filter import is_followed_ctrl
# from tailor.common.latest_utils import add_predictions_to_prompt_dict

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

    def run(
        self,
        processed_sentences: List[ProcessedSentence],
        generated_prompt_dicts: List[List[GeneratedPrompt]],
        spacy_model: SpacyModelType,
        srl_tagger: Optional[Predictor] = None,
        perplex_thred: Optional[int] = None,
        perplex_scorer: Optional[Tuple[PreTrainedModel, PreTrainedTokenizerBase]] = None,
    ) -> List[List[str]]:

        is_cuda = torch.cuda.is_available()

        if perplex_scorer:
            perplex_scorer = Munch(
                model=perplex_scorer[0], tokenizer=perplex_scorer[1]
            )  # backwards compatibility.
        else:
            perplex_scorer = load_perplex_scorer(is_cuda=is_cuda)

        # srl_tagger = get_srl_tagger()  # TODO

        all_sentences = []
        assert len(processed_sentences) == len(generated_prompt_dicts)
        for idx, sentence in enumerate(processed_sentences):
            prompt_dicts = generated_prompt_dicts[idx]

            orig_doc = sentence.spacy_doc

            validated_set = []
            for prompt_dict in prompt_dicts:
                prompt_dict = prompt_dict  # .prompt_dict

                generated = prompt_dict.sentence
                if generated in validated_set or generated.lower() == orig_doc.text.lower():
                    continue
                is_valid = True
                generated_doc = spacy_model(generated)
                if perplex_thred is not None:
                    eop = compute_edit_ops(orig_doc, generated_doc)
                    pp = compute_delta_perplexity(eop, perplex_scorer, is_cuda=is_cuda)
                    is_valid = pp.pr_sent < perplex_thred and pp.pr_phrase < perplex_thred
                # if is_valid:
                #     predicted = predict_batch_srl([generated.strip()], srl_tagger)[0]
                #     prompt_dict = add_predictions_to_prompt_dict(prompt_dict, predicted)
                #     is_valid = is_valid and is_followed_ctrl(prompt_dict, generated_doc, spacy_model)
                if is_valid:
                    validated_set.append(generated)
            all_sentences.append(validated_set)
        return all_sentences
