{
    "steps": {
        //"perturb": {"type": "simple-perturbation", "inputs": ["Hi this is a test", "Sample input text"]},
        "spacy_it": {
            "type": "process-with-spacy",
            "spacy_model_name": "en_core_web_sm",
            "sentences": ["Hi this is a test", "Sample input text"],
            //"step_format": "json",
        }
    }
}
