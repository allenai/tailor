from typing import List

from tango.step import Step

from tailor.common.abstractions import PromptObject


@Step.register("combine-all-prompts")
class CombineAllPrompts(Step):

    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, list_of_prompts: List[List[List[PromptObject]]]) -> List[List[PromptObject]]:

        num_sentences = [len(prompts) for prompts in list_of_prompts]
        assert len(set(num_sentences)) == 1  # All lists contain equal number of sentences.

        combined_prompts: List[List[PromptObject]] = [[] for _ in range(len(list_of_prompts[0]))]

        for prompts in list_of_prompts:
            for idx, sentence_prompts in enumerate(prompts):
                combined_prompts[idx] += sentence_prompts

        return combined_prompts
