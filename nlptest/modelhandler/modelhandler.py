from abc import ABC, abstractmethod
from typing import List

import spacy
from transformers import pipeline

from ..utils.custom_types import NEROutput, NERPrediction, SequenceClassificationOutput


class _ModelHandler(ABC):
    """Abstract base class for handling different models.

    Implementations should inherit from this class and override load_model() and predict() methods.
    """

    @abstractmethod
    def load_model(self):
        """Load the model.
        """
        return NotImplementedError()

    @abstractmethod
    def predict(self, text: str, *args, **kwargs):
        """Perform predictions on input text.
        """
        return NotImplementedError()


class ModelFactory:
    """
    A factory class for instantiating models.
    """
    SUPPORTED_TASKS = ["ner", "text-classification"]

    def __init__(
            self,
            model_path: str,
            task: str
    ):
        """Initializes the ModelFactory object.
        Args:
            model_path (str): path to model to use
            task (str): task to perform

        Raises:
            ValueError: If the task specified is not supported.
        """
        assert task in self.SUPPORTED_TASKS, \
            ValueError(f"Task '{task}' not supported. Please choose one of: {', '.join(self.SUPPORTED_TASKS)}")

        self.model_path = model_path
        self.task = task

        class_map = {
            cls.__name__.replace("PretrainedModel", "").lower(): cls for cls in _ModelHandler.__subclasses__()
        }
        if any([m in model_path for m in ["_core_news_", "_core_web_", "_ent_wiki_"]]):
            self.backend = "spacy"
        else:
            self.backend = "transformers"
        model_class_name = task.replace("-", "") + self.backend

        self.model_class = class_map[model_class_name](self.model_path)

    def load_model(self) -> None:
        """Load the model."""
        self.model_class.load_model()

    def predict(self, text: str, **kwargs) -> List[NEROutput]:
        """Perform predictions on input text.

        Args:
            text (str): Input text to perform predictions on.

        Returns:
            List[NEROutput]:
                List of NEROutput objects representing the entities and their corresponding labels.
        """
        return self.model_class(text=text, **kwargs)

    def __call__(self, text: str, *args, **kwargs) -> List[NEROutput]:
        """Alias of the 'predict' method"""
        return self.model_class(text=text, **kwargs)


class NERTransformersPretrainedModel(_ModelHandler):
    """
    Args:
        model_path (str):
            path to model to use
    """

    def __init__(
            self,
            model_path: str
    ):
        """
        Attributes:
            model_path (str):
                Path to the pretrained model to use.
            model (transformers.pipeline.Pipeline):
                Loaded NER pipeline for predictions.
        """
        self.model_path = model_path
        self.model = None

    def load_model(self) -> None:
        """Load the NER model into the `model` attribute.
        """
        self.model = pipeline(model=self.model_path, task="ner", ignore_labels=[])

    def predict(self, text: str, **kwargs) -> NEROutput:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform NER on.
            kwargs: Additional keyword arguments.

        Keyword Args:
            group_entities (bool): Option to group entities.

        Returns:
            List[NEROutput]: A list of named entities recognized in the input text.

        Raises:
            OSError: If the `model` attribute is None, meaning the model has not been loaded yet.
        """
        if self.model is None:
            raise OSError(f"The model '{self.model_path}' has not been loaded yet. Please call "
                          f"the '.load_model' method before running predictions.")
        prediction = self.model(text, **kwargs)

        prediction = [group for group in self.model.group_entities(prediction) if group["entity_group"] != "O"]

        return NEROutput(predictions=[NERPrediction(**pred) for pred in prediction])

    def __call__(self, text: str, *args, **kwargs) -> NEROutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text, **kwargs)


class NERSpaCyPretrainedModel(_ModelHandler):
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

    def load_model(self) -> None:
        """"""
        self.model = spacy.load(self.model_path)

    def predict(self, text: str, *args, **kwargs) -> NEROutput:
        """"""
        if self.model is None:
            raise OSError(f"The model '{self.model_path}' has not been loaded yet. Please call "
                          f"the '.load_model' method before running predictions.")
        text = text.lower()
        kwargs["group_entities"] = True
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


class TextClassificationTransformersPretrainedModel(_ModelHandler):
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

    def load_model(self) -> None:
        """"""
        self.model = pipeline(model=self.model_path, task="text-classification")

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


class TextClassificationSpacyPretrainedModel(_ModelHandler):
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
