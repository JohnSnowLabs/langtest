from transformers import pipeline, Pipeline

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

    def load_model(self, path) -> 'Pipeline':
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

        # prediction = [group for group in self.model.group_entities(prediction) if group["entity_group"] != "O"]
        return NEROutput(predictions=[NERPrediction.from_span(
            entity=pred.get('entity_group', pred.get('entity', None)),
            word=pred['word'],
            start=pred['start'],
            end=pred['end']
        ) for pred in prediction])

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
    def labels(self):
        """Return classification labels of pipeline model."""
        return list(self.model.model.config.id2label.values())

    def load_model(self, path) -> None:
        """Load and return text classification transformers pipeline"""
        self.model = pipeline(model=path, task="text-classification")

    def predict(self, text: str, return_all_scores: bool = False, *args, **kwargs) -> SequenceClassificationOutput:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform NER on.
            return_all_scores (bool): Option to group entities.
            kwargs: Additional keyword arguments.

        Returns:
            SequenceClassificationOutput: text classification from the input text.
        """
        if return_all_scores:
            kwargs["top_k"] = len(self.labels)

        output = self.model(text, **kwargs)
        return SequenceClassificationOutput(
            text=text,
            labels=output
        )

    def __call__(self, text: str, return_all_scores: bool = False, *args, **kwargs) -> SequenceClassificationOutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text, return_all_scores=return_all_scores, **kwargs)
