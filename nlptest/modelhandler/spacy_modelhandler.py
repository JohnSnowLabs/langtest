import spacy
from spacy.tokens import Doc

from .modelhandler import _ModelHandler
from ..utils.custom_types import NEROutput, NERPrediction, SequenceClassificationOutput


class PretrainedModelForNER(_ModelHandler):
    """
    Args:
        model: Pretrained SpaCy pipeline.
    """

    def __init__(
            self,
            model
    ):
        annotation = getattr(model, '__call__').__annotations__
        assert (annotation.get('return') and annotation['return'] is Doc), \
            ValueError(f"Invalid SpaCy Pipeline. Expected return type is {Doc} "
                       f"but pipeline returns: {annotation.get('return', None)}")

        self.model = model

    @classmethod
    def load_model(cls, path):
        """Load and return SpaCy pipeline"""
        return spacy.load(path)

    def predict(self, text: str, *args, **kwargs) -> NEROutput:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform NER on.
            kwargs: Additional keyword arguments.

        Keyword Args:
            group_entities (bool): Option to group entities.

        Returns:
            NEROutput: A list of named entities recognized in the input text.
        """
        doc = self.model(text)

        if kwargs.get("group_entities"):
            return NEROutput(
                predictions=[
                    NERPrediction.from_span(
                        entity=ent.label_,
                        word=ent.text,
                        start=ent.start_char,
                        end=ent.end_char
                    ) for ent in doc.ents
                ]
            )

    def __call__(self, text: str, *args, **kwargs) -> NEROutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text)


class PretrainedModelForTextClassification(_ModelHandler):
    """
    Args:
        model: Pretrained SpaCy pipeline.
    """

    def __init__(
            self,
            model
    ):
        annotation = getattr(model, '__call__').__annotations__
        assert (annotation.get('return') and annotation['return'] is Doc), \
            ValueError(f"Invalid SpaCy Pipeline. Expected return type is {Doc} "
                       f"but pipeline returns: {annotation.get('return', None)}")

        self.model = model

    @property
    def labels(self):
        """Return classification labels of SpaCy model."""
        return self.model.get_pipe("textcat").labels

    def load_model(self, path):
        """Load and return SpaCy pipeline"""
        return spacy.load(path)

    def predict(self, text: str, return_all_scores: bool = False, *args, **kwargs) -> SequenceClassificationOutput:
        """Perform text classication predictions on the input text.

        Args:
            text (str): Input text to classify.
            return_all_scores (bool): Option to return score for all labels.

        Returns:
            SequenceClassificationOutput: Text classification predictions from the input text.
        """
        output = self.model(text).cats
        if not return_all_scores:
            output = max(output, key=output.get)

        return SequenceClassificationOutput(
            text=text,
            labels=output
        )

    def __call__(self, text: str, return_all_scores: bool = False, *args, **kwargs) -> SequenceClassificationOutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text, return_all_scores=return_all_scores, **kwargs)
