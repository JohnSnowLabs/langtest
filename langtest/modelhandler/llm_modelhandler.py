import inspect
from typing import Any, Union
import langchain.llms as lc
import langchain.chat_models as cm
from langchain import LLMChain, PromptTemplate
from pydantic import ValidationError
from ..modelhandler.modelhandler import ModelAPI, LANGCHAIN_HUBS
from ..errors import Errors, Warnings
import logging
from functools import lru_cache
from langtest.utils.custom_types.helpers import HashableDict


class PretrainedModelForQA(ModelAPI):
    """A class representing a pretrained model for question answering.

    Attributes:
        model: The loaded pretrained model.
        hub: The hub name for the model.
        kwargs: Additional keyword arguments.

    Raises:
        ValueError: If the model is not found online or locally.
        ConfigError: If there is an error in the model configuration.
    """

    HUB_PARAM_MAPPING = {
        "azure-openai": "max_tokens",
        "ai21": "maxTokens",
        "cohere": "max_tokens",
        "openai": "max_tokens",
        "huggingface-inference-api": "max_length",
    }

    def __init__(self, hub: str, model: Any, *args, **kwargs):
        """Constructor class

        Args:
            hub (str): The hub name for the model.
            model (str): The model name or path.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.model = model
        self.hub = LANGCHAIN_HUBS[hub]
        self.kwargs = kwargs
        if isinstance(model, str):
            self.model = self.load_model(hub, model, *args, **kwargs).model

        self.predict.cache_clear()

    @classmethod
    def load_model(cls, hub: str, path: str, *args, **kwargs) -> "PretrainedModelForQA":
        """Load the pretrained model.

        Args:
            hub (str): The hub name for the model.
            path (str): The model path.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            PretrainedModelForQA: The loaded pretrained model.

        Raises:
            ValueError: If the model is not found online or locally.
            ConfigError: If there is an error in the model configuration.
        """
        try:
            kwargs.pop("task", None), kwargs.pop("device", None)
            cls._update_model_parameters(hub, kwargs)
            if path in ("gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview"):
                model = cm.ChatOpenAI(model=path, *args, **kwargs)
                return cls(hub, model, *args, **kwargs)
            else:
                model = getattr(lc, LANGCHAIN_HUBS[hub])
            default_args = inspect.getfullargspec(model).kwonlyargs
            if "model" in default_args:
                cls.model = model(model=path, *args, **kwargs)
            elif "model_name" in default_args:
                cls.model = model(model_name=path, *args, **kwargs)
            elif "model_id" in default_args:
                cls.model = model(model_id=path, *args, **kwargs)
            elif "repo_id" in default_args:
                cls.model = model(repo_id=path, model_kwargs=kwargs)
            return cls(hub, cls.model, *args, **kwargs)

        except ImportError:
            raise ValueError(Errors.E044.format(path=path))
        except ValidationError as e:
            error_msg = [err["loc"][0] for err in e.errors()]
            raise ConfigError(
                Errors.E045.format(
                    path=path, hub=hub, field=error_msg[0], error_message=e
                )
            )

    @classmethod
    def _update_model_parameters(cls, hub: str, kwargs: dict):
        """Update model parameters based on the hub's mapping.

        Args:
            hub (str): The hub name for the model.
            kwargs (dict): Keyword arguments to be updated.
        """
        if hub == "azure-openai" and "deployment_name" not in kwargs:
            kwargs["deployment_name"] = "text-davinci-003"
            logging.warning(Warnings.W014.format(hub=hub, kwargs=kwargs))

        if "max_tokens" in kwargs and hub in cls.HUB_PARAM_MAPPING:
            new_tokens_key = cls.HUB_PARAM_MAPPING[hub]
            kwargs[new_tokens_key] = kwargs.pop("max_tokens")

    @lru_cache(maxsize=102400)
    def predict(self, text: Union[str, dict], prompt: dict, *args, **kwargs):
        """Perform prediction using the pretrained model.

        Args:
            text (Union[str, dict]): The input text or dictionary.
            prompt (dict): The prompt configuration.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The prediction result.
        """
        try:
            prompt_template = PromptTemplate(**prompt)
            llmchain = LLMChain(prompt=prompt_template, llm=self.model)
            output = llmchain.run(**text)
            return output
        except Exception as e:
            raise ValueError(Errors.E089.format(error_message=e))

    def predict_raw(self, text: Union[str, dict], prompt: dict, *args, **kwargs):
        """Perform raw prediction using the pretrained model.

        Args:
            text (Union[str, dict]): The input text or dictionary.
            prompt (dict): The prompt configuration.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The prediction result.
        """
        return self.predict(text, prompt, *args, **kwargs)

    def __call__(self, text: Union[str, dict], prompt: dict, *args, **kwargs):
        """Perform prediction using the model.

        Args:
            text (Union[str, dict]): The input text or dictionary.
            prompt (dict): The prompt configuration.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The prediction result.
        """

        if isinstance(text, dict):
            text = HashableDict(**text)
        prompt = HashableDict(**prompt)
        return self.predict(text, prompt, *args, **kwargs)


class ConfigError(BaseException):
    """An exception raised for configuration errors.

    Args:
        message (str): The error message.

    Attributes:
        message (str): The error message.

    Examples:
        >>> raise ConfigError('Invalid configuration')
    """

    def __init__(self, message: str):
        """Constructor method

        Args:
             message (str): message to display
        """
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class PretrainedModelForSummarization(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for summarization.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForToxicity(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for toxicity detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForSecurity(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for security detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForClinicalTests(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for security detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForDisinformationTest(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for disinformation test.
    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForPolitical(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for security detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForSensitivityTest(PretrainedModelForQA, ModelAPI):
    def __init__(self, hub: str, model: Any, *args, **kwargs):
        """
        Initialize the PretrainedModelForSensitivityTest.

        Args:
            hub (str): The hub name associated with the pretrained model.
            model (Any): The pretrained model to be used.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(hub, model, *args, **kwargs)

    @lru_cache(maxsize=102400)
    def predict(self, text: Union[str, dict], *args, **kwargs):
        """Perform prediction using the pretrained model.

        Args:
            text (Union[str, dict]): The input text or dictionary.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the prediction result.
                - 'result': The prediction result.
        """
        try:
            prompt = PromptTemplate(input_variables=["text"], template="{text}")
            llmchain = LLMChain(prompt=prompt, llm=self.model)
            result = llmchain.run({"text": text})
            return {
                "result": result,
            }
        except Exception as e:
            raise ValueError(Errors.E089.format(error_message=e))

    def __call__(self, text: Union[str, dict], *args, **kwargs):
        """
        Alias of the 'predict' method.

        Args:
            text (Union[str, dict]): The original text or dictionary.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the prediction result.
                - 'result': The prediction result.
        """
        if isinstance(text, dict):
            text = HashableDict(**text)
        return self.predict(
            text=text,
            *args,
            **kwargs,
        )


class PretrainedModelForWinoBias(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for wino-bias detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForLegal(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for legal-tests.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForFactualityTest(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for factuality detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForSycophancyTest(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for sycophancy test.

    This class inherits from PretrainedModelForQA and provides functionality
    specific to sycophancy task.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass
