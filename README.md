# Tailor

This repository contains code for compositional perturbations as described in the following paper:

> [Tailor: Generating and Perturbing Text with Semantic Controls](https://arxiv.org/abs/2107.07150)  
> Alexis Ross*, Tongshuang Wu*, Hao Peng, Matthew E. Peters, Matt Gardner
> Association for Computational Linguistics (ACL), 2022

Bibtex for citations:

```bibtex
@inproceedings{ross-etal-2022-tailor,
    title = "Tailor: Generating and Perturbing Text with Semantic Controls",
    author = "Ross, Alexis and
        Wu, Tongshuang and
        Peng, Hao and
        Peters, Matthew E and
            Gardner, Matt",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    month = aug,
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```

## Installation

From Pypi:

```bash
pip install tailor_nlp
```

From source:

```bash
git clone https://github.com/allenai/tailor.git
cd tailor
pip install -e .
```

## Recreating Tailor
See [link](https://github.com/allenai/tailor/tree/main/scripts/train) for information on how to format Ontonotes 5.0 and train the Tailor generator.

## Tailor-Generated Contrast Sets
See [link](https://github.com/allenai/tailor/new/main/contrast_sets) for the data. (More information in Section 5 of the paper.)

## Using Tailor: walkthrough cases

- See the [tutorial notebook](https://github.com/allenai/tailor/blob/main/examples/Tutorial%2001%20-%20Tailor%20basic%20wrapper.ipynb) for a detailed walkthrough of the API.
- See the documents in the [main Python file](https://github.com/allenai/tailor/blob/main/tailor/tailor_wrapper.py) for more explanations.
- See [Tutorial 02](https://github.com/allenai/tailor/blob/main/examples/Tutorial%2002%20-%20Using%20Tailor%20default%20perturb.%20func%20with%20NLI.ipynb) to learn how to use the default perturbation function on NLI data.
- See [Tutorial 03](https://github.com/allenai/tailor/blob/main/examples/Tutorial%2003%20-%20Defining%20customized%20perturb.%20func%20for%20MATRES.ipynb) to learn how to define a customized perturbation function for MATRES data.

## Basic Perturbation demo

```py
# initiate a wrapper.
from tailor import Tailor
tl = Tailor()

text = "In the operation room, the doctor comforted the athlete."

# perturb the sentence with one line:
# When running it for the first time, the wrapper will automatically
# load related models, e.g. the generator and the perplexity filter.
perturbations = tl.perturb(text)

# return: [
# 'the athlete was comforted by the doctor .',
# 'In which case , the doctor comforted the athlete.',]
```

### More advanced APIs

To perturb with more controls,

```py
perturbations = tl.perturb(
    sentence=text,
    selected_span = "In the operation room",
    # can filter perturbations by their change type, as printed above.
    allowed_perturbs=["change_content"],
    # can reuse the detected strategies
    candidate_inputs = perturb_strategies,
    # filter out degeneration with gpt-2 perplexity score. If None, then this step is skiped.
    perplex_thred=50,
    # max number of perturbations to return.
    num_perturbs=10
)

# return: ["In case of an injury , the doctor 's comforted the athlete.",
# "In case of a fatal accident , the doctor 's comforted the athlete.",
# "In case of a bruised hand , the doctor 's comforted the athlete."]
```

To attach additional context,

```py
tl.perturb_with_context(
    "In the operation room, the doctor comforted the athlete.",
    "In the operation room",
    to_content="bridge",
    verbalize=True
)
# return: ["Under the bridge , the doctor 's comforted the athlete.",
# "Under a bridge , the doctor 's comforted the athlete."]

tl.perturb_with_context(
    "In the operation room, the doctor comforted the athlete.",
    "In the operation room",
    to_semantic_role="TEMPORAL",
    verbalize=True
)

# return: ['When the doctor came into the operation room , the physician comforted the athlete.',
# "While the doctor was in the operation room , the physician 's comforted the athlete."]


tl.perturb_with_context(
    "In the operation room, the doctor comforted the athlete.",
    "comforted",
    to_tense="future",
    verbalize=True
)

# return: ['In the operation room , the doctor will comfort the athlete.',
# "In the operation room , the doctor 's will comfort the athlete."]
```
