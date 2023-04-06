from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, Field, PrivateAttr, validator


class Span(BaseModel):
    """Representation of a text's slice"""
    start: int
    end: int
    word: str

    def shift_start(self, offset: int) -> None:
        """"""
        self.start -= offset

    def shift_end(self, offset: int) -> None:
        """"""
        self.end -= offset

    def shift(self, offset: int) -> None:
        """"""
        self.start -= offset
        self.end -= offset

    def __hash__(self):
        """"""
        return hash(self.__repr__())

    def __eq__(self, other):
        """"""
        return self.start == other.start and self.end == other.end

    def __str__(self):
        """"""
        return f"<Span(start={self.start}, end={self.end}, word='{self.word}')>"

    def __repr__(self):
        """"""
        return f"<Span(start={self.start}, end={self.end}, word='{self.word}')>"


class NERPrediction(BaseModel):
    """Single prediction obtained from a named entity recognition model"""
    entity: str = Field(None, alias="entity_group")
    span: Span
    score: Optional[float] = None
    doc_id: Optional[int] = None
    doc_name: Optional[str] = None
    pos_tag: Optional[str] = None
    chunk_tag: Optional[str] = None

    class Config:
        extra = "ignore"
        allow_population_by_field_name = True

    @classmethod
    def from_span(
            cls,
            entity: str,
            word: str,
            start: int,
            end: int,
            score: float = None,
            doc_id: int = None,
            doc_name: str = None,
            pos_tag: str = None,
            chunk_tag: str = None
    ) -> "NERPrediction":
        """"""
        return cls(
            entity=entity,
            span=Span(start=start, end=end, word=word),
            score=score,
            doc_id=doc_id,
            doc_name=doc_name,
            pos_tag=pos_tag,
            chunk_tag=chunk_tag
        )

    def __hash__(self):
        """"""
        return hash(self.__repr__())

    def __eq__(self, other):
        """"""
        if isinstance(other, NERPrediction):
            return self.entity == other.entity and \
                   self.span.start == other.span.start and \
                   self.span.end == other.span.end
        return False

    def __str__(self) -> str:
        """"""
        return f"{self.span.word}: {self.entity}"

    def __repr__(self) -> str:
        """"""
        return f"<NERPrediction(entity='{self.entity}', span={self.span})>"


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

    def __getitem__(self, item: Union[Span, int]) -> Optional[NERPrediction]:
        """"""
        if isinstance(item, int):
            return self.predictions[item]
        elif isinstance(item, Span):
            for prediction in self.predictions:
                if prediction.span == item:
                    return prediction
            return None

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


class SequenceLabel(BaseModel):
    """Single prediction obtained from text-classification models"""
    label: str
    score: float

    def __str__(self):
        """"""
        return f"{self.label}"


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


Result = TypeVar("Result", NEROutput, SequenceClassificationOutput, MinScoreOutput, MaxScoreOutput)


class Transformation(BaseModel):
    """
    Helper object keeping track of an alteration performed on a piece of text.
    It holds information about how a given span was transformed into another one
    """
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
    _realigned_spans: Optional[Result] = PrivateAttr(default_factory=None)
    category: str = None
    state: str = None

    # TODO: remove _realigned_spans, but for now it ensures that we don't realign spans multiple times

    def __init__(self, **data):
        super().__init__(**data)
        self._realigned_spans = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns the dict version of sample."""
        expected_result = self.expected_results.to_str_list()
        actual_result = self.actual_results.to_str_list() if self.actual_results is not None else None

        result = {
            'category': self.category,
            'test_type': self.test_type,
            'original': self.original,
            'test_case': self.test_case,
            'expected_result': expected_result,
        }

        if actual_result is not None:
            result.update({
                'actual_result': actual_result,
                'pass': self.is_pass()
            })

        return result

    @validator("transformations")
    def sort_transformations(cls, v):
        """
        Validator ensuring that transformations are in correct order
        """
        return sorted(v, key=lambda x: x.original_span.start)

    @property
    def ignored_predictions(self) -> List[NERPrediction]:
        """
        List of predictions that should be ignored because of the perturbations applied

        Returns:
            List[NERPrediction]: list of predictions which should be ignored
        """
        if not hasattr(self.actual_results, 'predictions'):
            return self.actual_results
        predictions = []

        for prediction in self.actual_results.predictions:
            for transformation in self.irrelevant_transformations:
                if transformation.new_span.start <= prediction.span.start \
                        and transformation.new_span.end >= prediction.span.end:
                    predictions.append(prediction)
        return predictions

    @property
    def relevant_transformations(self) -> Optional[List[Transformation]]:
        """
        Retrieves the transformations that need to be taken into account to realign `original` and `test_case`.

        Returns:
            Optional[List[Transformation]]: list of transformations which shouldn't be ignored
        """
        if not self.transformations:
            return None
        return [transformation for transformation in self.transformations if not transformation.ignore]

    @property
    def irrelevant_transformations(self) -> Optional[List[Transformation]]:
        """
        Retrieves the transformations that do not need to be taken into account to realign `original` and `test_case`.

        Returns:
            Optional[List[Transformation]]: list of transformations which should be ignored
        """
        if not self.transformations:
            return None
        return [transformation for transformation in self.transformations if transformation.ignore]

    @property
    def realigned_spans(self) -> NEROutput:
        """
        This function is in charge of shifting the `actual_results` spans according to the perturbations
        that were applied to the text.

        Note: we ignore predicted spans that were added during a perturbation

        Returns:
             NEROutput:
                realigned NER predictions
        """

        if self._realigned_spans is None:
            if len(self.transformations or '') == 0:
                return self.actual_results

            reversed_transformations = list(reversed(self.transformations))
            ignored_predictions = self.ignored_predictions

            realigned_results = []
            if hasattr(self.actual_results, "predictions"):
                for actual_result in self.actual_results.predictions:
                    if actual_result in ignored_predictions:
                        continue

                    for transformation in reversed_transformations:
                        if transformation.original_span.start == actual_result.span.start and \
                                transformation.new_span == actual_result.span:
                            # only the end of the span needs to be adjusted
                            actual_result.span.shift_end(transformation.new_span.end - transformation.original_span.end)
                        elif transformation.new_span.start < actual_result.span.start:
                            # the whole span needs to be shifted to the left
                            actual_result.span.shift(
                                (transformation.new_span.start - transformation.original_span.start) +
                                (transformation.new_span.end - transformation.original_span.end)
                            )
                        elif transformation.new_span.start >= actual_result.span.start and \
                                transformation.new_span.end <= actual_result.span.end:
                            # transformation nested in a span
                            actual_result.span.shift_end(
                                transformation.new_span.end - transformation.original_span.end
                            )

                    realigned_results.append(actual_result)

                self._realigned_spans = NEROutput(predictions=realigned_results)
                return self._realigned_spans
            else:
                return self.actual_results

        return self._realigned_spans

    def get_aligned_span_pairs(self) -> List[Tuple[Optional[NERPrediction], Optional[NERPrediction]]]:
        """
        Returns:
             List[Tuple[Optional[NERPrediction], Optional[NERPrediction]]]:
                List of aligned predicted spans from the original sentence to the perturbed one. The
                tuples are of the form: (perturbed span, original span). The alignment is achieved by
                using the transformations apply to the original text. If a Span couldn't be aligned
                with any other the tuple is of the form (Span, None) (or (None, Span)).
        """
        aligned_results = []
        expected_predictions_set, actual_predictions_set = set(), set()
        realigned_spans = self.realigned_spans

        # Retrieving and aligning perturbed spans for later comparison
        if self.relevant_transformations:
            for transformation in self.relevant_transformations:
                expected_prediction = self.expected_results[transformation.original_span]
                actual_prediction = realigned_spans[transformation.original_span]

                aligned_results.append((expected_prediction, actual_prediction))
                expected_predictions_set.add(expected_prediction)
                actual_predictions_set.add(actual_prediction)

        # Retrieving predictions for spans from the original sentence
        for expected_prediction in self.expected_results.predictions:
            if expected_prediction in expected_predictions_set:
                continue
            actual_prediction = realigned_spans[expected_prediction.span]
            aligned_results.append((expected_prediction, actual_prediction))
            expected_predictions_set.add(expected_prediction)
            if actual_prediction is not None:
                actual_predictions_set.add(actual_prediction)

        # Retrieving predictions for spans from the perturbed sentence
        for actual_prediction in realigned_spans.predictions:
            if actual_prediction in actual_predictions_set:
                continue
            expected_prediction = self.expected_results[actual_prediction.span]
            aligned_results.append((expected_prediction, actual_prediction))
            actual_predictions_set.add(actual_prediction)
            if expected_prediction is not None:
                expected_predictions_set.add(expected_prediction)

        return aligned_results

    def is_pass(self) -> bool:
        """"""
        if isinstance(self.actual_results, NEROutput):
            return all([a == b for (a, b) in self.get_aligned_span_pairs()])
        elif isinstance(self.actual_results, MinScoreOutput):
            return self.actual_results.min_score >= self.expected_results.min_score
        elif isinstance(self.actual_results, MaxScoreOutput):
            return self.actual_results.max_score <= self.expected_results.max_score
        else:
            filtered_actual_results = self.actual_results

        return self.expected_results == filtered_actual_results
