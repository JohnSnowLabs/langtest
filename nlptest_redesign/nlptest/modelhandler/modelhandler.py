from abc import ABC, abstractmethod
from typing import List

from transformers import pipeline

from ..utils.custom_types import NEROutput


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
    """A factory class for instantiating models.
    """
    SUPPORTED_TASKS = ["ner"]

    def __init__(
            self,
            model_path: str,
            task: str
    ):
        """ Initializes the ModelFactory object.
        Args:
            model_path (str): path to model to use
            task (str): task to perform

        Raises:
            ValueError: If the task specified is not supported.
        """
        assert task in self.SUPPORTED_TASKS, \
            ValueError(f"Task '{task}' not supported. Please choose one of {', '.join(self.SUPPORTED_TASKS)}")

        self.model_path = model_path
        self.task = task

        class_map = {
            cls.__name__.replace("PretrainedModel", "").lower(): cls for cls in _ModelHandler.__subclasses__()
        }
        self.model_class = class_map[self.task](self.model_path)

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


class NERPretrainedModel(_ModelHandler):
    """Model handler for pretrained NER models. Subclass of _ModelHandler.
    """

    def __init__(
            self,
            model_path: str
    ):
        """Initializes the NERPretrainedModel object.
        Args:
            model_path (str): Path to the pretrained model to use.

        Attributes:
            model_path (str): Path to the pretrained model to use.
            model (transformers.pipeline.Pipeline): Loaded NER pipeline for predictions.
        """
        self.model_path = model_path
        self.model = None

    def load_model(self) -> None:
        """Load the NER model into the `model` attribute.
        """
        self.model = pipeline(model=self.model_path, task="ner", ignore_labels=[])

    def predict(self, text: str, **kwargs) -> List[NEROutput]:
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
        prediction = self.model(text)

        if kwargs.get("group_entities"):
            prediction = [group for group in self.model.group_entities(prediction) if group["entity_group"] != "O"]

        return [NEROutput(**pred) for pred in prediction]

    def __call__(self, text: str, *args, **kwargs) -> List[NEROutput]:
        """Alias of the 'predict' method"""
        return self.predict(text=text, **kwargs)
