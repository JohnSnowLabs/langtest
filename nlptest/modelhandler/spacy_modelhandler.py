import spacy
from .modelhandler import _ModelHandler
from ..utils.custom_types import NEROutput, NERPrediction, SequenceClassificationOutput


class PretrainedModelForNER(_ModelHandler):
    """
    Args:
        model: Pretrained spacy model.
    """

    def __init__(
            self,
            model
    ):
        self.model = model

    @classmethod
    def load_model(cls, path) -> 'NERSpaCyPretrainedModel':
        """"""
        return cls(
            model=spacy.load(path)
        )

    def predict(self, text: str, *args, **kwargs) -> NEROutput:
        """"""
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
            model_path: str
    ):
        self.model_path = model_path
        self.model = None

    @property
    def labels(self):
        """"""
        return self.model.get_pipe("textcat").labels

    def load_model(self) -> None:
        """"""
        self.model = spacy.load(self.model_path)

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
