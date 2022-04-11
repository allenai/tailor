# Tailor-generated contrast sets for UD English dependency parsing. 

`noun2verb.csv` and `verb2noun.csv` contain perturbations changing prepositional phrase attachment types (noun/verb) for each instance in the original contrast sets created by [Gardner et al., 2021](https://arxiv.org/pdf/2004.02709.pdf). 

Each contrast set file contains the following fields:
- ``idx``: The unique index associated with a particular original instance
- ``original``: The original sentence
- ``human_perturbed``: Contrast set perturbation created by humans (taken from: [link](https://github.com/allenai/contrast-sets/tree/main/UD_English))
- ``tailor_perturbed``: Tailor-generated prompt
- ``pp_changed``: Prepositional phrase of ``original`` modified in perturbation (heuristically extracted from parse of original sentence)
- ``valid?``: Manual annotation of the validity of ``tailor-perturbed``. Possible values: 
  * `Y`: Valid (changes prepositional phrase in desired way)
  * `N`: Invalid (does *not* change prepositional phrase in desired way)
  * `M`: Valid with minor modifictions
  * `--`: Skipped due to a failure of perturbation strategy (i.e. Tailor produced no outputs)
  * `skip`: Skipped since a valid Tailor-generated perturbation for the same instance was already found
- ``tailor_perturbed_prompt``: Prompt used to source perturbed generation from Tailor

**Note**:  In calculating top-k validity (k=10), we source k generations for each original instance (to which the perturbation strategies apply). However, we only annotate a subset of these k generations efficiency purposes: We stop when we find one valid perturbation for an instance.
