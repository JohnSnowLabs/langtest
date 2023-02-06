from typing import List, TypeVar

from pydantic import BaseModel, Field


class NERPrediction(BaseModel):
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
    label: str
    score: float


class SequenceClassificationOutput(BaseModel):
    text: str
    labels: List[SequenceLabel]

    def __str__(self):
        labels = {elt.label: elt.score for elt in self.labels}
        return f"SequenceClassificationOutput(text='{self.text}', labels={labels})"

    def __eq__(self, other):
        """"""
        top_class = max(self.labels, key=lambda x: x.score).label
        other_top_class = max(other.labels, key=lambda x: x.score).label
        return top_class == other_top_class


Result = TypeVar("Result", List[NEROutput], SequenceClassificationOutput)


class Sample(BaseModel):
    test_case: str
    original: str
    expected_results: Result
    actual_results: Result

    def is_pass(self) -> bool:
        """"""
        return self.expected_results == self.actual_results
