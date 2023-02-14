from typing import List, Optional, TypeVar, Tuple

from pydantic import BaseModel, Field


class Span(BaseModel):
    start: int
    end: int
    word: str

    def __hash__(self):
        """"""
        return hash(self.__repr__())

    def __eq__(self, other):
        """"""
        return self.start == other.start and \
               self.end == other.end and \
               self.word == other.word

    def __repr__(self):
        """"""
        return f"<Span(start={self.start}, end={self.end}, word='{self.word}')>"


class NERPrediction(BaseModel):
    """"""
    entity: str = Field(None, alias="entity_group")
    span: Span
    score: Optional[float] = None

    class Config:
        extra = "ignore"
        allow_population_by_field_name = True

    @classmethod
    def from_span(cls, entity: str, word: str, start: int, end: int, score: float = None) -> "NERPrediction":
        """"""
        return cls(
            entity=entity,
            span=Span(start=start, end=end, word=word),
            score=score
        )

    def __hash__(self):
        """"""
        return hash(self.__repr__())

    def __eq__(self, other):
        """"""
        if isinstance(other, NERPrediction):
            return self.entity == other.entity
        return False

    def __repr__(self):
        """"""
        return f"<NERPrediction(entity='{self.entity}', span={self.span})>"


class NEROutput(BaseModel):
    """
    Output model for NER tasks.
    """
    predictions: List[NERPrediction]

    def __len__(self):
        """"""
        return len(self.predictions)

    def __getitem__(self, span: Span) -> Optional[NERPrediction]:
        """"""
        for pred in self.predictions:
            if pred.span == span:
                return pred
        return None

    def __eq__(self, other: "NEROutput"):
        """"""
        # NOTE: we need the list of transformations applied to the sample to be able
        # to align and compare two different NEROutput
        raise NotImplementedError()


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
    original_span: Span
    new_span: Span
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
    def relevant_transformations(self):
        """"""
        return [transformation for transformation in self.transformations if not transformation.ignore]

    @property
    def aligned_spans(self) -> List[Tuple[Optional[NERPrediction], Optional[NERPrediction]]]:
        """
        Returns:
             List[Tuple[Optional[NERPrediction], Optional[NERPrediction]]]:
                List of aligned predicted spans from the original sentence to the perturbed one. The
                tuples are of the form: (perturbed span, original span). The alignment is achieved by
                using the transformations apply to the original text. If a Span couldn't be aligned
                with any other the tuple is of the form (Span, None) (or (None, Span)).
        """
        aligned_results = []
        expected_pred_set, actual_pred_set = set(), set()

        # Retrieving and aligning perturbed spans for later comparison
        for transformation in self.relevant_transformations:
            expected_pred = self.expected_results[transformation.original_span]
            actual_pred = self.actual_results[transformation.new_span]

            aligned_results.append((expected_pred, actual_pred))
            expected_pred_set.add(expected_pred)
            actual_pred_set.add(actual_pred)

        # Retrieving predictions for spans that were not perturbed
        for expected_pred in self.expected_results.predictions:
            if expected_pred in expected_pred_set:
                continue
            actual_pred = self.actual_results[expected_pred.span]
            aligned_results.append((expected_pred, actual_pred))
            expected_pred_set.add(expected_pred)
            if actual_pred is not None:
                actual_pred_set.add(actual_pred)

        for actual_pred in self.actual_results.predictions:
            if actual_pred in actual_pred_set:
                continue
            expected_pred = self.actual_results[actual_pred.span]
            aligned_results.append((expected_pred, actual_pred))
            actual_pred_set.add(actual_pred)
            if expected_pred is not None:
                expected_pred_set.add(expected_pred)

        return aligned_results

    def is_pass(self) -> bool:
        """"""
        if isinstance(self.actual_results, NEROutput):
            return all([a == b for (a, b) in self.aligned_spans])
        else:
            filtered_actual_results = self.actual_results

        return self.expected_results == filtered_actual_results
