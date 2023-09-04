import importlib
from abc import ABC, abstractmethod
from typing import List, Union

from langtest.utils.lib_manager import try_import_lib
from ..utils.custom_types import NEROutput, SequenceClassificationOutput

RENAME_HUBS = {
    "azureopenai": "azure-openai",
    "huggingfacehub": "huggingface-inference-api",
}

if try_import_lib("langchain"):
    import langchain

    LANGCHAIN_HUBS = {
        RENAME_HUBS.get(hub.lower(), hub.lower())
        if hub.lower() in RENAME_HUBS
        else hub.lower(): hub
        for hub in langchain.llms.__all__
    }
else:
    LANGCHAIN_HUBS = {}


class _ModelHandler(ABC):
    """Abstract base class for handling different models.

    Implementations should inherit from this class and override load_model() and predict() methods.
    """

    @abstractmethod
    def load_model(cls, *args, **kwargs):
        """Load the model."""
        raise NotImplementedError()

    @abstractmethod
    def predict(self, text: Union[str, dict], *args, **kwargs):
        """Perform predictions on input text."""
        raise NotImplementedError()


class ModelFactory:
    """A factory class for instantiating models."""

    SUPPORTED_TASKS = [
        "ner",
        "text-classification",
        "question-answering",
        "summarization",
        "toxicity",
        "translation",
        "security",
        "clinical-tests",
        "disinformation-test",
        "political",
    ]
    SUPPORTED_MODULES = [
        "pyspark",
        "sparknlp",
        "nlu",
        "transformers",
        "spacy",
        "langchain",
    ]
    SUPPORTED_HUBS = [
        "spacy",
        "huggingface",
        "johnsnowlabs",
        "openai",
        "cohere",
        "ai21",
    ] + list(LANGCHAIN_HUBS.keys())

    def __init__(self, model: str, task: str, hub: str, *args, **kwargs):
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
        assert task in self.SUPPORTED_TASKS, ValueError(
            f"Task '{task}' not supported. Please choose one of: {', '.join(self.SUPPORTED_TASKS)}"
        )

        module_name = model.__module__.split(".")[0]
        assert module_name in self.SUPPORTED_MODULES, ValueError(
            f"Module '{module_name}' is not supported. "
            f"Please choose one of: {', '.join(self.SUPPORTED_MODULES)}"
        )

        if module_name in ["pyspark", "sparknlp", "nlu"]:
            model_handler = importlib.import_module(
                "langtest.modelhandler.jsl_modelhandler"
            )

        elif module_name == "langchain":
            model_handler = importlib.import_module(
                "langtest.modelhandler.llm_modelhandler"
            )

        else:
            model_handler = importlib.import_module(
                f"langtest.modelhandler.{module_name}_modelhandler"
            )

        if task == "ner":
            self.model_class = model_handler.PretrainedModelForNER(model)
        elif task in ("question-answering"):
            _ = kwargs.pop("user_prompt") if "user_prompt" in kwargs else kwargs
            self.model_class = model_handler.PretrainedModelForQA(
                hub, model, *args, **kwargs
            )
        elif task in ("summarization"):
            _ = kwargs.pop("user_prompt") if "user_prompt" in kwargs else kwargs
            self.model_class = model_handler.PretrainedModelForSummarization(
                hub, model, *args, **kwargs
            )
        elif task in ("toxicity"):
            _ = kwargs.pop("user_prompt") if "user_prompt" in kwargs else kwargs
            self.model_class = model_handler.PretrainedModelForToxicity(
                hub, model, *args, **kwargs
            )

        elif task in ("clinical-tests"):
            _ = kwargs.pop("user_prompt") if "user_prompt" in kwargs else kwargs
            self.model_class = model_handler.PretrainedModelForClinicalTests(
                hub, model, *args, **kwargs
            )

        elif task in ("disinformation-test"):
            _ = kwargs.pop("user_prompt") if "user_prompt" in kwargs else kwargs
            self.model_class = model_handler.PretrainedModelForDisinformationTest(
                hub, model, *args, **kwargs
            )

        elif task == "political":
            _ = kwargs.pop("user_prompt") if "user_prompt" in kwargs else kwargs
            self.model_class = model_handler.PretrainedModelForPolitical(
                hub, model, *args, **kwargs
            )

        elif task == "translation":
            self.model_class = model_handler.PretrainedModelForTranslation(model)

        elif task == "security":
            self.model_class = model_handler.PretrainedModelForSecurity(
                hub, model, *args, **kwargs
            )

        else:
            self.model_class = model_handler.PretrainedModelForTextClassification(model)

    @classmethod
    def load_model(
        cls, task: str, hub: str, path: str, *args, **kwargs
    ) -> "ModelFactory":
        """Loads the model.

        Args:
            path (str):
                path to model to use
            task (str):
                task to perform
            hub (str):
                model hub to load custom model from the path, either to hub or local disk.

        Returns:
            ModelHandler:
        """
        assert task in cls.SUPPORTED_TASKS, ValueError(
            f"Task '{task}' not supported. Please choose one of: {', '.join(cls.SUPPORTED_TASKS)}"
        )

        assert hub in cls.SUPPORTED_HUBS, ValueError(
            f"Invalid 'hub' parameter. Supported hubs are: {', '.join(cls.SUPPORTED_HUBS)}"
        )

        if hub == "johnsnowlabs":
            if importlib.util.find_spec("johnsnowlabs"):
                modelhandler_module = importlib.import_module(
                    "langtest.modelhandler.jsl_modelhandler"
                )
            else:
                raise ModuleNotFoundError(
                    """Please install the johnsnowlabs library by calling `pip install johnsnowlabs`.
                For in-depth instructions, head-over to https://nlu.johnsnowlabs.com/docs/en/install"""
                )

        elif hub == "huggingface":
            if importlib.util.find_spec("transformers"):
                modelhandler_module = importlib.import_module(
                    "langtest.modelhandler.transformers_modelhandler"
                )
            else:
                raise ModuleNotFoundError(
                    """Please install the transformers library by calling `pip install transformers`.
                For in-depth instructions, head-over to https://huggingface.co/docs/transformers/installation"""
                )

        elif hub == "spacy":
            if importlib.util.find_spec("spacy"):
                modelhandler_module = importlib.import_module(
                    f"langtest.modelhandler.{hub}_modelhandler"
                )
            else:
                raise ModuleNotFoundError(
                    """Please install the spacy library by calling `pip install spacy`.
                For in-depth instructions, head-over to https://spacy.io/usage"""
                )

        elif hub.lower() in LANGCHAIN_HUBS:
            modelhandler_module = importlib.import_module(
                "langtest.modelhandler.llm_modelhandler"
            )

        if task == "ner":
            model_class = modelhandler_module.PretrainedModelForNER.load_model(path)
        elif task in ("question-answering"):
            _ = kwargs.pop("user_prompt") if "user_prompt" in kwargs else kwargs
            model_class = modelhandler_module.PretrainedModelForQA.load_model(
                hub, path, *args, **kwargs
            )
        elif task in ("summarization"):
            _ = kwargs.pop("user_prompt") if "user_prompt" in kwargs else kwargs
            model_class = modelhandler_module.PretrainedModelForSummarization.load_model(
                hub, path, *args, **kwargs
            )
        elif task in ("toxicity"):
            _ = kwargs.pop("user_prompt") if "user_prompt" in kwargs else kwargs
            model_class = modelhandler_module.PretrainedModelForToxicity.load_model(
                hub, path, *args, **kwargs
            )
        elif task == "translation":
            model_class = modelhandler_module.PretrainedModelForTranslation.load_model(
                path
            )

        elif task == "security":
            model_class = modelhandler_module.PretrainedModelForSecurity.load_model(
                hub, path, *args, **kwargs
            )

        elif task == "clinical-tests":
            model_class = modelhandler_module.PretrainedModelForClinicalTests.load_model(
                hub, path, *args, **kwargs
            )
        elif task == "political":
            model_class = modelhandler_module.PretrainedModelForPolitical.load_model(
                hub, path, *args, **kwargs
            )

        elif task in ("disinformation-test"):
            model_class = (
                modelhandler_module.PretrainedModelForDisinformationTest.load_model(
                    hub, path, *args, **kwargs
                )
            )

        else:
            model_class = (
                modelhandler_module.PretrainedModelForTextClassification.load_model(path)
            )

        return cls(model_class, task, hub, *args, **kwargs)

    def predict(
        self, text: Union[str, dict], **kwargs
    ) -> Union[NEROutput, SequenceClassificationOutput]:
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

    def __call__(
        self, text: Union[str, dict], *args, **kwargs
    ) -> Union[NEROutput, SequenceClassificationOutput]:
        """Alias of the 'predict' method

        Args:
            text (str): Input text to perform predictions on.

        Returns:
            Union[NEROutput, SequenceClassificationOutput]:
                predicted output
        """
        return self.model_class(text=text, **kwargs)
