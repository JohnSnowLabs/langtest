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
            List[NEROutput]: A list of named entities recognized in the input text.
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
    Args:
        model_path (str):
            path to model to use
    """

    def __init__(
            self,
            model_path: str
    ):
        self.model_path = model_path
        self.model = None

    @property
    def labels(self):
        """"""
        return list(self.model.model.config.id2label.values())

    def load_model(self, path) -> None:
        """"""
        self.model = pipeline(model=path, task="text-classification")

    def predict(self, text: str, return_all_scores: bool = False, *args, **kwargs) -> SequenceClassificationOutput:
        """"""
        if return_all_scores:
            kwargs["top_k"] = len(self.labels)

        output = self.model(text, **kwargs)
        return SequenceClassificationOutput(
            text=text,
            labels=output
        )

    def __call__(self, text: str, return_all_scores: bool = False, *args, **kwargs) -> SequenceClassificationOutput:
        """"""
        return self.predict(text=text, return_all_scores=return_all_scores, **kwargs)

