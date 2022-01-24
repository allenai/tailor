from tango.common.registrable import Registrable


class PerturbFunction(Registrable):
    def __call__(self, *args, **kwargs):  # TODO: maybe fix args?
        raise NotImplementedError


class PerturbStringFunction(Registrable):
    def __call__(self, prompt_meta, *args, **kwargs) -> str:
        raise NotImplementedError


@PerturbStringFunction.register("change_voice")
class ChangeVoice(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> str:

        vtense = prompt_meta.vtense
        target_voice = "active" if prompt_meta.vvoice is "passive" else "passive"

        perturb_str = (
            f"CONTEXT(DELETE_TEXT),VERB(CHANGE_TENSE({vtense}),CHANGE_VOICE({target_voice}))"
        )
        # return "VERB(CHANGE_VOICE())"
        return perturb_str


@PerturbStringFunction.register("change_tense")
class ChangeTense(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> str:
        return "VERB(CHANGE_TENSE())"


@PerturbStringFunction.register("change_lemma")
class ChangeLemma(PerturbStringFunction):
    def __call__(self, prompt_meta, lemma: str, *args, **kwargs) -> str:  # type: ignore
        return f"VERB(CHANGE_LEMMA({lemma}))"


@PerturbStringFunction.register("delete_text")
class DeleteText(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> str:
        return "CONTEXT(DELETE_TEXT)"


@PerturbStringFunction.register("delete_punctuation")
class DeletePunctuation(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> str:
        return "CONTEXT(DELETE_PUNCT)"


@PerturbStringFunction.register("swap_core_with_context")
class SwapCoreWithContext(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> str:
        return "CORE(SWAP_CORE)"


@PerturbStringFunction.register("swap_core_without_context")
class SwapCoreWithoutContext(PerturbStringFunction):
    def __call__(self, prompt_meta, *args, **kwargs) -> str:
        return "CONTEXT(DELETE_TEXT),CORE(SWAP_CORE)"
