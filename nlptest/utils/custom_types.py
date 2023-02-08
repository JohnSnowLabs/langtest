from typing import List, TypeVar

from pydantic import BaseModel, Field


class NERPrediction(BaseModel):
    """"""
    entity: str = Field(None, alias="entity_group")
    word: str
    start: int
    end: int
    score: float = None
    ignore: bool = False

    class Config:
        extra = "ignore"
        allow_population_by_field_name = True

    def __eq__(self, other):
        return self.entity == other.entity and \
               self.start == other.score and \
               self.end == other.end


class NEROutput(BaseModel):
    """
    Output model for NER tasks.
    """
    labels: List[NERPrediction]

    def __eq__(self, other):
        """"""
        non_ignored_preds = [pred for pred in self.labels if not pred.ignore]
        non_ignored_preds_other = [pred for pred in other.labels if not pred.ignore]

        if len(non_ignored_preds) != len(non_ignored_preds_other):
            return SyntaxError(f"Cannot compare NEROutputs as one has {len(non_ignored_preds)} tokens and the other "
                               f"{len(non_ignored_preds_other)}.")

        labels = sorted(non_ignored_preds, key=lambda x: x.start)
        other_labels = sorted(non_ignored_preds_other, key=lambda x: x.start)
        return all([label == label_other for label, label_other in zip(labels, other_labels)])


class SequenceLabel(BaseModel):
    """"""
    label: str
    score: float


class SequenceClassificationOutput(BaseModel):
    """
    Output model for text classification tasks.
    """
    labels: List[SequenceLabel]

    def __str__(self):
        labels = {elt.label: elt.score for elt in self.labels}
        return f"SequenceClassificationOutput(labels={labels})"

    def __eq__(self, other):
        """"""
        top_class = max(self.labels, key=lambda x: x.score).label
        other_top_class = max(other.labels, key=lambda x: x.score).label
        return top_class == other_top_class


Result = TypeVar("Result", List[NEROutput], SequenceClassificationOutput)


class Sample(BaseModel):
    """
    Helper object storing the original text, the perturbed one and the corresponding
    predictions for each of them.

    The specificity here is that it is task-agnostic, one only needs to call access the `is_pass`
    property to assess whether the `expected_results` and the `actual_results` are the same, regardless
    the downstream task.

    This way, to support a new task one only needs to create a `XXXOutput` model, overload the `__eq__`
    operator and add the new model to the `Result` type variable.
    """
    original: str
    test_type: str
    test_case: str = None
    expected_results: Result = None
    actual_results: Result = None

    def is_pass(self) -> bool:
        """"""
        return self.expected_results == self.actual_results
