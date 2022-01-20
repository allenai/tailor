from tailor.steps.process_with_spacy import *
from tailor.steps.get_srl_tags import *
from tailor.steps.generate_prompts_by_tags import *

if __name__ == "__main__":
    # sentence = "The doctor comforted the athlete."
    sentence = "The athlete who was seen by the judges called the manager."
    # verbs: was, seen, called.
    p = ProcessWithSpacy()
    spacy_outs = p.run(sentences=[sentence])
    p = SRLTags()
    processed_sents = p.run(spacy_outs)
    p = GeneratePromptsByTags()
    p.run(processed_sents, keyword_str="EXACT,UNCASED", nblanks=10)
