
local premises = ["The doctor comforted the patient.", "The book was picked by the girl."];
{
    "steps": {
        /*"raw_data": {
            "type": "datasets::load",
            "path": "snli",
            "split": "train",
        },
        "premises": {
            "type": "get-sentences",
            "dataset": {"type": "ref", "ref": "raw_data"},
            "key": "premise",
            "start_idx": 1000,
            "end_idx": 1100,
        },*/
        "spacy_model": {
            "type": "get-spacy-model",
            //"parse": true,
        },
        "premises_spacy": {
            "type": "process-with-spacy",
            "spacy_model": {"type": "ref", "ref": "spacy_model"},
            //"sentences": {"type": "ref", "ref": "premises"},
            "sentences": premises,
            "cache_results": false, //we store the spacy outputs in srl.
        },
        "premises_srl": {
            "type": "get-srl-tags",
            "spacy_outputs": {"type": "ref", "ref": "premises_spacy"},
        },
        "intermediate_prompts": {
            "type": "generate-prompts-by-tags",
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "keyword_str": "EXACT,UNCASED",
            "nblanks": 10,
        },
        /*"premise_perturbations": {
            "type": "perturb-prompt-with-intermediate",
            "intermediate_prompts": {"type": "ref", "ref": "intermediate_prompts"},
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "perturb_fn": "change_voice",
        },*/
        /*"premise_perturbations": {
            "type": "perturb-prompt",
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "intermediate_prompt_kwargs": {
                "keyword_str": "EXACT,UNCASED",
                "nblanks": 10,
            },
            "perturb_fn": "change_voice_single",
        },*/
        /*"premise_perturbations": {
            "type": "perturb-prompt-by-str",
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "intermediate_prompt_kwargs": {
                "keyword_str": "EXACT,UNCASED",
                "nblanks": null,
            },
            "perturb_str": "VERB(CHANGE_VOICE())",
        },*/
        "premise_perturbations": {
            "type": "perturb-prompt-by-str-with-intermediate",
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "intermediate_prompts": {"type": "ref", "ref": "intermediate_prompts"},
            "perturb_str": "VERB(CHANGE_VOICE())",
        },
        "generated_premises": {
            "type": "generate-from-prompts",
            "prompts": {"type": "ref", "ref": "premise_perturbations"},
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "spacy_model": {"type": "ref", "ref": "spacy_model"},
        }
    }   
}