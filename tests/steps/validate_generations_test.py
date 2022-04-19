from munch import Munch

from tailor.common.abstractions import GeneratedPrompt
from tailor.common.testing import TailorTestCase
from tailor.steps.validate_generations import ValidateGenerations


class TestValidateGenerations(TailorTestCase):
    def setup_method(self):
        super().setup_method()

        self.generated_prompts = [
            [
                GeneratedPrompt(
                    prompt_no_header="- [PATIENT: the patient] is [VERB: comforted], [AGENT : by the doctor]",
                    sentence="- the patient is comforted , by the doctor",
                    clean_sentence="the patient is comforted, by the doctor",
                    meta=None,
                    annotations=None,
                    words=None,
                    vidx=4,
                    name=None,
                    perplexities=Munch(
                        {"pr_sent": 20.57333755493164, "pr_phrase": 29.992721557617188}
                    ),
                ),
                GeneratedPrompt(
                    prompt_no_header="Having's [VERB: comforted] - [PATIENT: the patient] [AGENT : by the doctor]",
                    sentence="Having 's comforted - the patient by the doctor",
                    clean_sentence="Having's comforted-the patient by the doctor",
                    meta=None,
                    annotations=None,
                    words=None,
                    vidx=2,
                    name=None,
                    perplexities=Munch(
                        {"pr_sent": 15.902446746826172, "pr_phrase": 20.792152404785156}
                    ),
                ),
                GeneratedPrompt(
                    prompt_no_header="- [PATIENT: the patient] was [VERB: comforted] [AGENT; by the doctor].",
                    sentence="- the patient was comforted by the doctor .",
                    clean_sentence="the patient was comforted by the doctor.",
                    meta=None,
                    annotations=None,
                    words=None,
                    vidx=4,
                    name=None,
                    perplexities=Munch(
                        {"pr_sent": 14.877056121826172, "pr_phrase": 10.63360595703125}
                    ),
                ),
            ],
            [
                GeneratedPrompt(
                    prompt_no_header="- [PATIENT: the book]'s [VERB: picked] [AGENT : by the girl]",
                    sentence="- the book 's picked by the girl",
                    sentence="the book's picked by the girl",
                    meta=None,
                    annotations=None,
                    words=None,
                    vidx=4,
                    name=None,
                    perplexities=Munch(
                        {"pr_sent": 15.22769546508789, "pr_phrase": -9.461502075195312}
                    ),
                ),
            ],
        ]

    def test_step(self):
        step = ValidateGenerations()

        result = step.run(
            generated_prompt_dicts=self.generated_prompts,
            perplex_thresh=20,
        )

        assert len(result) == 2

        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert isinstance(result[0][0], str)
