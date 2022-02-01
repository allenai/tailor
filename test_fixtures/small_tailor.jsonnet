
local premises = [
    "The doctor comforted the patient.", 
    "The book was picked by the girl.",
    "A little boy drinks milk and gets milk all over his face and table."
];

{
    "steps": {
        "spacy_model": {
            "type": "get-spacy-model",
            "parse": true,
        },
        "premises_spacy": {
            "type": "process-with-spacy",
            "spacy_model": {"type": "ref", "ref": "spacy_model"},
            "sentences": premises,
            //"cache_results": false, //we store the spacy outputs in srl.
        },
        "premises_srl": {
            "type": "get-srl-tags",
            "spacy_outputs": {"type": "ref", "ref": "premises_spacy"},
        },
        "premise_perturbations_voice": {
            "type": "perturb-prompt-with-str",
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "perturb_str_func": "change_voice",
            /*"intermediate_prompt_kwargs": {
                "nblanks": 10,
            },*/
        },
        "premise_perturbations_shorten_core": {
            "type": "perturb-prompt-with-str",
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "intermediate_prompt_kwargs": {
                "nblanks": 10,
            },
            "perturb_str_func": "shorten_core_argument"
        },
        "premise_perturbations": {
            "type": "combine-all-prompts",
            "list_of_prompts": [
                {"type": "ref", "ref": "premise_perturbations_voice"},
                {"type": "ref", "ref": "premise_perturbations_shorten_core"},
            ]
        },
        "generated_premises_dicts": {
            "type": "generate-from-prompts",
            "prompts": {"type": "ref", "ref": "premise_perturbations"},
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "spacy_model": {"type": "ref", "ref": "spacy_model"},
        },
        "generated_premises": {
            "type": "validate-generations",
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "generated_prompt_dicts": {"type": "ref", "ref": "generated_premises_dicts"},
            "spacy_model": {"type": "ref", "ref": "spacy_model"},
        }
    }   
}