
local premises = ["The doctor comforted the patient.", "The book was picked by the girl."];

local perturb_func = "change_voice";

{
    "steps": {
        "spacy_model": {
            "type": "get-spacy-model",
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
        "premise_perturbations": {
            "type": "perturb-prompt-with-str",
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "intermediate_prompt_kwargs": {
                "keyword_str": "EXACT,UNCASED",
                "nblanks": null,
            },
            //"perturb_str_func": perturb_func,
            "perturb_str_func": "CONTEXT(DELETE_TEXT),VERB(CHANGE_VOICE(passive))",
            //"perturb_str_func": "some_random_string",
        },
        "generated_premises": {
            "type": "generate-from-prompts",
            "prompts": {"type": "ref", "ref": "premise_perturbations"},
            "processed_sentences": {"type": "ref", "ref": "premises_srl"},
            "spacy_model": {"type": "ref", "ref": "spacy_model"},
        }
    }   
}