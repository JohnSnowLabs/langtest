from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union
from functools import lru_cache
from langtest.utils.lib_manager import try_import_lib

RENAME_HUBS = {
    "azureopenai": "azure-openai",
    "huggingfacehub": "huggingface-inference-api",
    "transformers": "huggingface",
    "jsl": "johnsnowlabs",
}

if try_import_lib("langchain"):
    import langchain
    import langchain.llms

    LANGCHAIN_HUBS = {
        RENAME_HUBS.get(hub.lower(), hub.lower())
        if hub.lower() in RENAME_HUBS
        else hub.lower(): hub
        for hub in langchain.llms.__all__
    }
else:
    LANGCHAIN_HUBS = {}


class ModelAPI(ABC):
    """Abstract base class for handling different models.

    Implementations should inherit from this class and override load_model() and predict() methods.
    """

    model_registry = defaultdict(lambda: defaultdict(lambda: ModelAPI))

    @abstractmethod
    def load_model(cls, *args, **kwargs):
        """Load the model."""
        raise NotImplementedError()

    @abstractmethod
    @lru_cache(maxsize=102400)
    def predict(self, text: Union[str, dict], *args, **kwargs):
        """Perform predictions on input text."""
        raise NotImplementedError()

    def __init_subclass__(cls, *args, **kwargs) -> None:
        hub = cls.__module__.split(".")[-1].split("_")[0]
        if hub in RENAME_HUBS:
            hub = RENAME_HUBS[hub]
        task = cls.__name__.replace("PretrainedModelFor", "").lower()
        ModelAPI.model_registry[hub][task] = cls
        return super().__init_subclass__(*args, **kwargs)
