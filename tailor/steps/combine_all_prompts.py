from typing import List

from tango.step import Step

from tailor.common.abstractions import PromptObject


@Step.register("combine-all-prompts")
class CombineAllPrompts(Step):
    """
    This step simply combines multiple lists of prompt objects generated from different
    perturbations into a flattened list of prompt objects with one list for each input
    sentence.

    list_of_prompts = [
        # prompts per sentence from perturbation 1
        [
            [PromptObject(...), PromptObject(...)], # sentence 1
            [PromptObject(...)], # sentence 2
            [], # sentence 3
            [PromptObject(...), PromptObject(...)], # sentence 4
        ],
        # prompts per sentence from perturbation 2
        [
            [PromptObject(...)], # sentence 1
            [], # sentence 2
            [PromptObject(...)], # sentence 3
            [PromptObject(...), PromptObject(...)], # sentence 4
        ]
    ]

    results in

    combined_prompts = [
        [
            [PromptObject(...), PromptObject(...), PromptObject(...)], # sentence 1
            [PromptObject(...)], # sentence 2
            [PromptObject(...)], # sentence 3
            [PromptObject(...), PromptObject(...), PromptObject(...), PromptObject(...)], # sentence 4
        ]
    ]

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "combine-all-prompts".
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, list_of_prompts: List[List[List[PromptObject]]]) -> List[List[PromptObject]]:
        """
        Parameters
        ----------

        list_of_prompts : List[List[List[:class:`PromptObject`]]]
            A list of prompts generated from different perturbations. See output
            of :class:`PerturbPromptWithString` or :class:`PerturbPromptWithFunction`
            for what each list item looks like.

        Returns
        -------

        List[List[:class:`PromptObject`]]
            The flattened list of prompts combining all perturbations.
        """
        num_sentences = [len(prompts) for prompts in list_of_prompts]
        assert (
            len(set(num_sentences)) == 1
        ), "All lists need to contain prompts for equal number of sentences."

        combined_prompts: List[List[PromptObject]] = [[] for _ in range(len(list_of_prompts[0]))]

        for prompts in list_of_prompts:
            for idx, sentence_prompts in enumerate(prompts):
                combined_prompts[idx] += sentence_prompts

        return combined_prompts
