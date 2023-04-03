import importlib
from abc import ABC, abstractmethod
from typing import List, Union

from ..utils.custom_types import NEROutput, SequenceClassificationOutput


class _ModelHandler(ABC):
    """
    Abstract base class for handling different models.

    Implementations should inherit from this class and override load_model() and predict() methods.
    """

    @abstractmethod
    def load_model(cls, path: str):
        """Load the model."""
        return NotImplementedError()

    @abstractmethod
    def predict(self, text: str, *args, **kwargs):
        """Perform predictions on input text."""
        return NotImplementedError()


class ModelFactory:
    """
    A factory class for instantiating models.
    """

    SUPPORTED_TASKS = ["ner", "text-classification"]
    SUPPORTED_MODULES = ['pyspark', 'sparknlp', 'nlu', 'transformers', 'spacy']
    SUPPORTED_HUBS = ['johnsnowlabs', 'spacy', 'huggingface']

    def __init__(
            self,
            model: str,
            task: str,
    ):
        """Initializes the ModelFactory object.
        Args:
            model (str):
                path to the model to evaluate
            task (str):
                task to perform

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

        if task == 'ner':
            self.model_class = model_handler.PretrainedModelForNER(model)
        else:
            self.model_class = model_handler.PretrainedModelForTextClassification(model)

    @classmethod
    def load_model(cls, task: str, hub: str, path: str) -> 'ModelFactory':
        """Loads the model.

        Args:
            path (str):
                path to model to use
            task (str):
                task to perform
            hub (str):
                model hub to load custom model from the path, either to hub or local disk.

        Returns
        """

        assert task in cls.SUPPORTED_TASKS, \
            ValueError(f"Task '{task}' not supported. Please choose one of: {', '.join(cls.SUPPORTED_TASKS)}")

        assert hub in cls.SUPPORTED_HUBS, \
            ValueError(f"Invalid 'hub' parameter. Supported hubs are: {', '.join(cls.SUPPORTED_HUBS)}")

        if hub == 'johnsnowlabs':
            if importlib.util.find_spec('johnsnowlabs'):
                modelhandler_module = importlib.import_module('nlptest.modelhandler.jsl_modelhandler')
            else:
                raise ModuleNotFoundError("""Please install the johnsnowlabs library by calling `pip install johnsnowlabs`.
                For in-depth instructions, head-over to https://nlu.johnsnowlabs.com/docs/en/install""")
            
        elif hub == 'huggingface':
            if importlib.util.find_spec('transformers'):
                modelhandler_module = importlib.import_module('nlptest.modelhandler.transformers_modelhandler')
            else:
                raise ModuleNotFoundError("""Please install the transformers library by calling `pip install transformers`.
                For in-depth instructions, head-over to https://huggingface.co/docs/transformers/installation""")
            
        elif hub == "spacy":
            if importlib.util.find_spec('spacy'):
                modelhandler_module = importlib.import_module(f'nlptest.modelhandler.{hub}_modelhandler')
            else:
                raise ModuleNotFoundError("""Please install the spacy library by calling `pip install spacy`.
                For in-depth instructions, head-over to https://spacy.io/usage""")

        if task == 'ner':
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
            Union[NEROutput, SequenceClassificationOutput]:
                predicted output
        """
        return self.model_class(text=text, **kwargs)

    def predict_raw(self, text: str) -> List[str]:
        """Perform predictions on input text.

        Args:
            text (str): Input text to perform predictions on.

        Returns:
            List[str]: Predictions.
        """
        return self.model_class.predict_raw(text)

    def __call__(self, text: str, *args, **kwargs) -> Union[NEROutput, SequenceClassificationOutput]:
        """Alias of the 'predict' method

        Args:
            text (str): Input text to perform predictions on.

        Returns:
            Union[NEROutput, SequenceClassificationOutput]:
                predicted output
        """
        return self.model_class(text=text, **kwargs)
