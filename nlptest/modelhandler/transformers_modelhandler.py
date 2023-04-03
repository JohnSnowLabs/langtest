from typing import List

from transformers import Pipeline, pipeline

from .modelhandler import _ModelHandler
from ..utils.custom_types import NEROutput, NERPrediction, SequenceClassificationOutput


class PretrainedModelForNER(_ModelHandler):
    """
    Args:
        model (transformers.pipeline.Pipeline): Pretrained HuggingFace NER pipeline for predictions.
    """

    def __init__(
            self,
            model
    ):
        """
        Attributes:
            model (transformers.pipeline.Pipeline):
                Loaded NER pipeline for predictions.
        """

        assert isinstance(model, Pipeline), \
            ValueError(f"Invalid transformers pipeline! "
                       f"Pipeline should be '{Pipeline}', passed model is: '{type(model)}'")
        self.model = model

    @classmethod
    def load_model(cls, path) -> 'Pipeline':
        """Load the NER model into the `model` attribute."""
        return pipeline(model=path, task="ner", ignore_labels=[])

    def predict(self, text: str, **kwargs) -> NEROutput:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform NER on.
            kwargs: Additional keyword arguments.

        Keyword Args:
            group_entities (bool): Option to group entities.

        Returns:
            NEROutput: A list of named entities recognized in the input text.
        """
        prediction = self.model(text, **kwargs)

        return NEROutput(predictions=[NERPrediction.from_span(
            entity=pred.get('entity_group', pred.get('entity', None)),
            word=pred['word'],
            start=pred['start'],
            end=pred['end']
        ) for pred in prediction])

    def predict_raw(self, text: str) -> List[str]:
        """
        Predict a list of labels.
        Args:
            text (str): Input text to perform NER on.
        Returns:
            List[str]: A list of named entities recognized in the input text.
        """
        prediction = self.model(text)
        return [x["entity"] for x in prediction]

    def __call__(self, text: str, *args, **kwargs) -> NEROutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text, **kwargs)


class PretrainedModelForTextClassification(_ModelHandler):
    """
    Attributes:
        model (transformers.pipeline.Pipeline):
            Loaded Text Classification pipeline for predictions.
    """

    def __init__(
            self,
            model,
    ):
        assert isinstance(model, Pipeline), \
            ValueError(f"Invalid transformers pipeline! "
                       f"Pipeline should be '{Pipeline}', passed model is: '{type(model)}'")
        self.model = model

    @property
    def labels(self) -> List[str]:
        """Return classification labels of pipeline model."""
        return list(self.model.model.config.id2label.values())

    @classmethod
    def load_model(cls, path: str) -> "Pipeline":
        """Load and return text classification transformers pipeline"""
        return pipeline(model=path, task="text-classification")

    def predict(self, text: str, return_all_scores: bool = False, truncation_strategy: str = "longest_first", *args,
                **kwargs) -> SequenceClassificationOutput:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform NER on.
            return_all_scores (bool): Option to group entities.
            truncation_strategy (str): strategy to use to truncate too long sequences
            kwargs: Additional keyword arguments.

        Returns:
            SequenceClassificationOutput: text classification from the input text.
        """
        if return_all_scores:
            kwargs["top_k"] = len(self.labels)

        output = self.model(text, truncation_strategy=truncation_strategy, **kwargs)
        return SequenceClassificationOutput(
            text=text,
            predictions=output
        )

    def predict_raw(self, text: str, truncation_strategy: str = "longest_first") -> List[str]:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform NER on.
            truncation_strategy (str): strategy to use to truncate too long sequences

        Returns:
            List[str]: Predictions as a list of strings.
        """
        return [pred["label"] for pred in self.model(text, truncation_strategy=truncation_strategy)]

    def __call__(self, text: str, return_all_scores: bool = False, *args, **kwargs) -> SequenceClassificationOutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text, return_all_scores=return_all_scores, **kwargs)
