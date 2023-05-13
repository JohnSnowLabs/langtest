from typing import List, Optional, TypeVar, Union

from pydantic import BaseModel, validator

from .helpers import Span
from .predictions import NERPrediction, SequenceLabel


class SequenceClassificationOutput(BaseModel):
    """
    Output model for text classification tasks.
    """
    predictions: List[SequenceLabel]

    def to_str_list(self) -> str:
        """Convert the output into list of strings.

        Returns:
            List[str]: predictions in form of a list of strings.
        """
        return ",".join([x.label for x in self.predictions])

    def __str__(self):
        """"""
        labels = {elt.label: elt.score for elt in self.predictions}
        return f"SequenceClassificationOutput(predictions={labels})"

    def __eq__(self, other):
        """"""
        top_class = max(self.predictions, key=lambda x: x.score).label
        other_top_class = max(other.predictions, key=lambda x: x.score).label
        return top_class == other_top_class


class MinScoreOutput(BaseModel):
    """Output for accuracy/representation tests."""
    min_score: float

    def to_str_list(self) -> float:
        """"""
        return self.min_score

    def __repr__(self) -> str:
        """"""
        return f"{self.min_score}"

    def __str__(self) -> str:
        """"""
        return f"{self.min_score}"


class MaxScoreOutput(BaseModel):
    """Output for accuracy/representation tests."""
    max_score: float

    def to_str_list(self) -> float:
        """"""
        return self.max_score

    def __repr__(self) -> str:
        """"""
        return f"{self.max_score}"

    def __str__(self) -> str:
        """"""
        return f"{self.max_score}"


class NEROutput(BaseModel):
    """
    Output model for NER tasks.
    """
    predictions: List[NERPrediction]

    @validator("predictions")
    def sort_by_appearance(cls, v):
        """"""
        return sorted(v, key=lambda x: x.span.start)

    def __len__(self):
        """"""
        return len(self.predictions)

    def __getitem__(self, item: Union[Span, int]) -> Optional[Union[List[NERPrediction], NERPrediction]]:
        """"""
        if isinstance(item, int):
            return self.predictions[item]
        elif isinstance(item, Span):
            for prediction in self.predictions:
                if prediction.span == item:
                    return prediction
            return None
        elif isinstance(item, slice):
            return [self.predictions[i] for i in range(item.start, item.stop)]

    def to_str_list(self) -> str:
        """
        Converts predictions into a list of strings.

        Returns:
            List[str]: predictions in form of a list of strings.
        """
        return ", ".join([str(x) for x in self.predictions if str(x)[-3:] != ': O'])

    def __repr__(self) -> str:
        """"""
        return self.predictions.__repr__()

    def __str__(self) -> str:
        """"""
        return [str(x) for x in self.predictions].__repr__()

    def __eq__(self, other: "NEROutput"):
        """"""
        # NOTE: we need the list of transformations applied to the sample to be able
        # to align and compare two different NEROutput
        raise NotImplementedError()


Result = TypeVar("Result", NEROutput, SequenceClassificationOutput, MinScoreOutput, MaxScoreOutput, List[str], str)
