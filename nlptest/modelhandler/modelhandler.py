import importlib
from abc import ABC, abstractmethod
from typing import List, Union

from ..utils.custom_types import NEROutput, SequenceClassificationOutput


class _ModelHandler(ABC):
    """Abstract base class for handling different models.

    Implementations should inherit from this class and override load_model() and predict() methods.
    """

    @abstractmethod
    def load_model(cls, path):
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
    SUPPORTED_MODULES = ['pyspark', 'sparknlp', 'nlu', 'transformers', 'spacy']
    SUPPORTED_HUBS = ['johnsnowlabs', 'spacy', 'transformers']

    def __init__(
            self,
            model,
            task: str,
    ):
        """Initializes the ModelFactory object.
        Args:
            model: SparkNLP, HuggingFace or Spacy model to test.
            task (str): task to perform

        Raises:
            ValueError: If the task specified is not supported.
            ValueError: If the given model is not supported.
        """
        assert task in self.SUPPORTED_TASKS, \
            ValueError(f"Task '{task}' not supported. Please choose one of: {', '.join(self.SUPPORTED_TASKS)}")

        module_name = model.__module__.split('.')[0]
        assert module_name in self.SUPPORTED_MODULES, \
            ValueError(f"Module '{module_name}' is not supported. "
                       f"Please choose one of: {', '.join(self.SUPPORTED_MODULES)}")

        if module_name in ['pyspark', 'sparknlp', 'nlu']:
            model_handler = importlib.import_module(f'nlptest.modelhandler.jsl_modelhandler')
        else:
            model_handler = importlib.import_module(f'nlptest.modelhandler.{module_name}_modelhandler')

        if task is 'ner':
            self.model_class = model_handler.PretrainedModelForNER(model)
        else:
            self.model_class = model_handler.PretrainedModelForTextClassification(model)

    @classmethod
    def load_model(
            cls,
            task: str,
            hub: str,
            path: str
    ) -> 'ModelFactory':
        """Load the model.

        Args:
            path (str): path to model to use
            task (str): task to perform
            hub (str): model hub to load custom model from the path, either to hub or local disk.
        """

        assert task in cls.SUPPORTED_TASKS, \
            ValueError(f"Task '{task}' not supported. Please choose one of: {', '.join(cls.SUPPORTED_TASKS)}")

        assert hub in cls.SUPPORTED_HUBS, \
            ValueError(f"Invalid 'hub' parameter. Supported hubs are: {', '.join(cls.SUPPORTED_HUBS)}")

        if hub is 'johnsnowlabs':
            modelhandler_module = importlib.import_module('nlptest.modelhandler.jsl_modelhandler')
        else:
            modelhandler_module = importlib.import_module(f'nlptest.modelhandler.{hub}_modelhandler')

        if task is 'ner':
            model_class = modelhandler_module.PretrainedModelForNER.load_model(path)
        else:
            model_class = modelhandler_module.PretrainedModelForTextClassification.load_model(path)

        return cls(
            model_class,
            task
        )

    def predict(self, text: str, **kwargs) -> Union[NEROutput, SequenceClassificationOutput]:
        """Perform predictions on input text.

        Args:
            text (str): Input text to perform predictions on.

        Returns:
            NEROutput or SequenceClassificationOutput
        """
        return self.model_class(text=text, **kwargs)

    def __call__(self, text: str, *args, **kwargs) -> List[NEROutput]:
        """Alias of the 'predict' method"""
        return self.model_class(text=text, **kwargs)
