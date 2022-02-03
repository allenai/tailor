from typing import List

from tango.common.registrable import Registrable

from tailor.common.utils.head_prompt_utils import get_unique_tags


class PerturbationCriteria(Registrable):
    def __call__(self, processed_sentence, *args, **kwargs):
        raise NotImplementedError


@PerturbationCriteria.register("all_verbs")
class AllVerbs(PerturbationCriteria):
    def __call__(self, processed_sentence, *args, **kwargs) -> List[List[str]]:
        return processed_sentence.get_tags_list()


class ArgsToBlankCondition(Registrable):  # TODO: rename this to something more user-friendly.
    def __call__(self, tags: List[List[str]], *args, **kwargs):
        raise NotImplementedError


@ArgsToBlankCondition.register("unique_tags")
class UniqueTags(ArgsToBlankCondition):
    def __call__(self, tags: List[List[str]], *args, **kwargs) -> List[List[str]]:
        return get_unique_tags(tags)
