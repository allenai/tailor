# AI2 Tailor

This is the codebase for the [Tailor](https://api.semanticscholar.org/CorpusID:235898904) library.

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
pip install polyjuice_nlp
```

From source:

```bash
git clone git@github.com:allenai/ai2-tailor.git
cd ai2-tailor
pip install -e .
```

Tailor depends on [SpaCy](https://spacy.io/) and [Huggingface Transformers](https://huggingface.co/). To use most functions, please also install the following:

```bash
# install pytorch, as here: https://pytorch.org/get-started/locally/#start-locally
pip install torch
```

## Features

- Generate random perturbations for input data.
- Build your own task-specific perturbations.

## Examples

Example notebooks can be found [here](examples/)
