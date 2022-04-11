# Tailor-generated contrast sets for UD English dependency parsing. 

We use Tailor to recreate the UD English contrast sets created by [Gardner et al., 2021](https://arxiv.org/pdf/2004.02709.pdf) ([data](https://github.com/allenai/contrast-sets/tree/main/UD_English)), which perturb sentences to change prepositional phrase attachment types (noun/verb).

## Data

`noun2verb.csv` and `verb2noun.csv` contain Tailor-generated perturbations for each instance in the original contrast sets. Each file contains the following fields:
- ``idx``: The unique index associated with a particular original instance
- ``original``: The original sentence
- ``human_perturbed``: Human-created perturbation
- ``tailor_perturbed``: Tailor-generated perturbation
- ``pp_changed``: Prepositional phrase of ``original`` whose attachment is modified (heuristically extracted from parse of original sentence)
- ``valid?``: Manual annotation of the validity of ``tailor-perturbed``. Possible values: 
  * `Y`: Valid (changes prepositional phrase in desired way)
  * `N`: Invalid (does *not* change prepositional phrase in desired way)
  * `M`: Valid with minor modifictions
  * `--`: Skipped due to a failure of perturbation strategy (i.e. Tailor produced no outputs)
  * `skip`: Skipped since a valid Tailor-generated perturbation for the same instance was already found
- ``tailor_perturbed_prompt``: Prompt used to source perturbed generation from Tailor

**Note**:  In calculating top-k validity (k=10), we source k generations for each original instance. However, we only annotate a subset of these k generations efficiency purposes: We stop when we find one valid perturbation for an instance.

For more information, see Section 5 of the [paper](https://arxiv.org/pdf/2107.07150v2.pdf).

## Analysis
`parsing_analysis.ipynb` contains code for computing the **top-k validity** (Section 5.1) and **lexical diversity** (Section 5.2) of the Tailor-generated contrast sets for UD English.
