from abc import ABC, abstractmethod
import importlib
from typing import List, Union
from ..utils.custom_types import NEROutput, SequenceClassificationOutput


class _ModelHandler(ABC):
    """Abstract base class for handling different models.

    Implementations should inherit from this class and override load_model() and predict() methods.
    """

    @classmethod
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
    SUPPORTED_MODULES = ['pyspark', 'sparknlp', 'sparknlp_jsl', 'nlu', 'transformers', 'spacy']

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
        """
        assert task in self.SUPPORTED_TASKS, \
            ValueError(f"Task '{task}' not supported. Please choose one of: {', '.join(self.SUPPORTED_TASKS)}")

        module_name = model.__module__.split('.')[0]
        assert module_name in self.SUPPORTED_MODULES, \
            ValueError(f"Module '{module_name}' is not supported. "
                       f"Please choose one of: {', '.join(self.SUPPORTED_MODULES)}")

        if module_name in ['pyspark', 'sparknlp', 'sparknlp_jsl', 'nlu']:
            model_handler = importlib.import_module('nlptest.nlptest.modelhandler.jsl_modelhandler')

        else:
            model_handler = importlib.import_module(f'nlptest.nlptest.modelhandler.{module_name}_modelhandler')

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
            hub (str): model hub to load custom model from the path
        """

        model_class_name = task + hub
        class_map = {
            cls.__name__.replace("PretrainedModel", "").lower(): cls for cls in _ModelHandler.__subclasses__()
        }

        model_class = class_map[model_class_name].load_model(path)
        return cls(
            model_class,
            task
        )

    def predict(self, text: str, **kwargs) -> Union[NEROutput, SequenceClassificationOutput]:
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