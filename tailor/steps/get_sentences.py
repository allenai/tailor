from typing import Any, Dict, Iterable, List, Optional, Union

import datasets
from tango.step import Step


@Step.register("get-sentences")
class GetSentences(Step):
    """
    Given a :class:`datasets.Dataset`, it returns a specific text field.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "get-sentences".
    """

    DETERMINISTIC = True
    CACHEABLE = True

    def run(
        self,
        dataset: Union[datasets.Dataset, Dict[str, List[Any]]],
        key: str,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Iterable[str]:
        """
        Returns a specific text field as an iterable.

        Parameters
        ----------

        dataset : :class:`datasets.Dataset`
            The dataset object from which to extract the text field.
        key : :class:`str`
            The name of the field to extract.
        start_idx : :class:`int`, optional
            Optionally, specify a starting index to extract a subset of all instances.
        end_idx : :class:`int`, optional
            Optionally, specify an ending index to extract a subset of all instances.

        Returns
        -------

        :class:`Iterable[str]`
            The specific text field as an iterable.
        """

        assert len(dataset[key]) > 0 and isinstance(
            dataset[key][0], str
        ), f"The field '{key}' is not a text field; this may lead to errors downstream."
        return dataset[key][start_idx:end_idx]  # None works as expected.
