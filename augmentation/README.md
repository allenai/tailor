#  Data augmentation with Tailor for Natural Language Inference

We use Tailor to create perturbations used in data augmentation for NLI (Section 6 of the paper).

## Data

`filtered_swap_core.csv` contains the augmented examples generated with Tailor for the SNLI training dataset. We perturb original SNLI hypotheses by applying the SWAP\_CORE perturbation strategy. We then filter generations by perplexity such that we retain the top 75% of generations in terms of fluency.

In our augmentation experiments, we treat original hypotheses as premises (i.e. ``new_premise``) and perturbed hypotheses as hypotheses (i.e. ``new_hypothesis``) in our augmentation experiments. For more details, see Section 6/Appendix E of the paper. 
