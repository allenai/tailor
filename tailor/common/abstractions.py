from munch import Munch


class Prompt(Munch):  # TODO: later change this to named tuple and make it specific.
    pass


class CriteriaForPerturbation:
    """
    TODO: add details
    A lot of the perturbation strategies might have criteria that must be met
    for the perturbation to be applied, and these criteria are task-specific,
    so we’ll want the user to supply them. A given sentence will have multiple
    prompts (one for each predicate), and we often only want to apply the perturbations
    for specific predicates and if some argument(s) contain some linguistic phenomenon
    (eg. prepositional phrases, a particular verb voice, etc.)
    """

    pass
