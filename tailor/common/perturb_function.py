from tango.common.registrable import Registrable


class PerturbFunction(Registrable):
    def __call__(self, *args, **kwargs):  # TODO: maybe fix args?
        return NotImplementedError
