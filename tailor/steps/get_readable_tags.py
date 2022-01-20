# from typing import Dict, Iterable, List
# from tango.step import Step

# @Step.register("get-readable-tags")
# class GeneratePromptsByTags(Step):
#     DETERMINISTIC = True
#     CACHEABLE = True

#     def run(
#         self,
#         srl_tags: Iterable[List[Dict]],
#     ) -> Iterable[List[List[str]]]:
#         """
#         Arguments:
#             srl_tags: List of tags for each verb in each sentence.
#                       Expected format is the output of `GetSRLTags` step.
#         """
#         convert_tag2readable(vlemma, raw_tag, frameset_id,
#         role_dicts=None,
#         frameset_path=DEFAULT_FRAME_SET_PATH):
