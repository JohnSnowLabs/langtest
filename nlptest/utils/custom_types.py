from typing import List, Optional, TypeVar

from pydantic import BaseModel, Field


class NERPrediction(BaseModel):
    """"""
    entity: str = Field(None, alias="entity_group")
    word: str
    start: int
    end: int
    score: Optional[float] = None
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
    predictions: List[NERPrediction]

    def __eq__(self, other):
        """"""
        non_ignored_preds = [pred for pred in self.predictions if not pred.ignore]
        non_ignored_preds_other = [pred for pred in other.predictions if not pred.ignore]

        if len(non_ignored_preds) != len(non_ignored_preds_other):
            return SyntaxError(f"Cannot compare NEROutputs as one has {len(non_ignored_preds)} tokens and the other "
                               f"{len(non_ignored_preds_other)}.")

        predictions = sorted(non_ignored_preds, key=lambda x: x.start)
        other_predictions = sorted(non_ignored_preds_other, key=lambda x: x.start)
        return all([label == label_other for label, label_other in zip(predictions, other_predictions)])


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


Result = TypeVar("Result", NEROutput, SequenceClassificationOutput)


class Transformation(BaseModel):
    from_start_char: int
    to_start_char: int
    from_end_char: int
    to_end_char: int
    ignore: bool = False


class Sample(BaseModel):
    """
    Helper object storing the original text, the perturbed one and the corresponding
    predictions for each of them.

    The specificity here is that it is task-agnostic, one only needs to call access the `is_pass`
    property to assess whether the `expected_results` and the `actual_results` are the same, regardless
    the downstream task.nlptest/utils/custom_types.py

    This way, to support a new task one only needs to create a `XXXOutput` model, overload the `__eq__`
    operator and add the new model to the `Result` type variable.
    """
    original: str
    test_type: str = None
    test_case: str = None
    expected_results: Result = None
    actual_results: Result = None
    transformations: List[Transformation] = None

    @property
    def filtered_results(self) -> NEROutput:
        """"""
        filtered_results = []

        for pred in self.actual_results.predictions:
            for transfo in self.transformations:
                if pred.start == transfo.to_start_char and (transfo.to_end_char - pred.end) <= 1 and transfo.ignore:
                    continue
                filtered_results.append(pred)
        return NEROutput(predictions=filtered_results)

    def is_pass(self) -> bool:
        """"""
        if isinstance(self.actual_results, NEROutput) and len(self.transformations) > 0:
            filtered_actual_results = self.filtered_results
        else:
            filtered_actual_results = self.actual_results
        return self.expected_results == filtered_actual_results
