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

    def load_model(self, path):
        """Load the SpaCy pipeline into the `model` attribute."""
        return spacy.load(path)

    def predict(self, text: str, *args, **kwargs) -> NEROutput:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform NER on.
            kwargs: Additional keyword arguments.

        Keyword Args:
            group_entities (bool): Option to group entities.

        Returns:
            List[NEROutput]: A list of named entities recognized in the input text.
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
        model_path (str):
            path to model to use
    """

    def __init__(
            self,
            model: str
    ):
        annotation = getattr(model, '__call__').__annotations__
        assert (annotation.get('return') and annotation['return'] is Doc), \
            ValueError(f"Invalid SpaCy Pipeline. Expected return type is {Doc} "
                       f"but pipeline returns: {annotation.get('return', None)}")

        self.model = model

    @property
    def labels(self):
        """"""
        return self.model.get_pipe("textcat").labels

    def load_model(self, path):
        """"""
        return spacy.load(path)

    def predict(self, text: str, return_all_scores: bool = False, *args, **kwargs) -> SequenceClassificationOutput:
        """"""
        output = self.model(text).cats
        if not return_all_scores:
            output = max(output, key=output.get)

        return SequenceClassificationOutput(
            text=text,
            labels=output
        )

    def __call__(self, text: str, return_all_scores: bool = False, *args, **kwargs) -> SequenceClassificationOutput:
        """"""
        return self.predict(text=text, return_all_scores=return_all_scores, **kwargs)
