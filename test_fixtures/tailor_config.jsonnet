
{
    "steps": {
        "raw_data": {
            "type": "datasets::load",
            "path": "snli",
            "split": "train",
            //"cache_results": true, // can't.
        },
        "premises": {
            "type": "get-sentences",
            "dataset": {"type": "ref", "ref": "raw_data"},
            "key": "premise",
            "start_idx": 1000,
            "end_idx": 1100,
        },
        "spacy_model": {
            "type": "get-spacy-model",
        },
        "premises_spacy": {
            "type": "process-with-spacy",
            "spacy_model": {"type": "ref", "ref": "spacy_model"},
            "sentences": {"type": "ref", "ref": "premises"},
        },
        "premises_srl": {
            "type": "srl-tags",
            "spacy_outputs": {"type": "ref", "ref": "premises_spacy"},
        },
        /*"intermediate_prompts": {
            "type": "generate-prompts-by-tags",
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "keyword_str": "EXACT,UNCASED",
            "nblanks": 10,
        },
        "premise_perturbations": {
            "type": "perturb-prompt-with-intermediate",
            "intermediate_prompts": {"type": "ref", "ref": "intermediate_prompts"},
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "perturb_fn": "change_voice",
        }*/
        "premise_perturbations": {
            "type": "perturb-prompt",
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "intermediate_prompt_kwargs": {
                "keyword_str": "EXACT,UNCASED",
                "nblanks": 10,
                "return_prompt_type": "concrete",
            },
            "perturb_fn": "change_voice_single",
        },
        /*"generated_premises": {
            "type": "generate-from-prompts",
            "prompts": "premise_perturbations",
            "processed_sentences": "premises_srl", //I think just spacy outs is needed?
        }*/
    }   
}