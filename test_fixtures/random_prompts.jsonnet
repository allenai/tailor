
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
        "common_keywords_by_tag": {
            "type": "get-common-keywords-by-tag",
        },
        "premise_perturbations": {
            "type": "generate-random-prompts",
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "common_keywords_by_tag": {"type": "ref", "ref": "common_keywords_by_tag"},
        },
        "generated_premises_dicts": {
            "type": "generate-from-prompts",
            "prompts": {"type": "ref", "ref": "premise_perturbations"},
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "spacy_model": {"type": "ref", "ref": "spacy_model"},
            "compute_perplexity": true,
        },
        "generated_premises": {
            "type": "validate-generations",
            //"processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "generated_prompt_dicts": {"type": "ref", "ref": "generated_premises_dicts"},
            //"spacy_model": {"type": "ref", "ref": "spacy_model"},
            "perplex_thresh": 10000, // simply to see all values.
        }
    }   
}