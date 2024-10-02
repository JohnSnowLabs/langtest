import inspect
from typing import Any, List, Union
import langchain.llms as lc
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic import Field, ValidationError

from langtest.utils.custom_types.output import NEROutput
from langtest.utils.custom_types.predictions import NERPrediction
from ..modelhandler.modelhandler import ModelAPI, LANGCHAIN_HUBS
from ..errors import Errors, Warnings
import logging
from functools import lru_cache
from langtest.utils.custom_types.helpers import HashableDict
from langchain.chat_models.base import BaseChatModel


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
        exclude_args = ["task", "device", "stream"]

        filtered_kwargs = kwargs.copy()

        for arg in exclude_args:
            filtered_kwargs.pop(arg, None)

        try:
            cls._update_model_parameters(hub, filtered_kwargs)
            if path in (
                "gpt-4o-mini",
                "gpt-4",
                "gpt-3.5-turbo",
                "gpt-4-1106-preview",
                "gpt-4-turbo-2024-04-09",
                "gpt-4-0125-preview",
                "gpt-3.5-turbo-0125",
                "gpt-4-turbo-preview",
                "gpt-4-vision-preview",
                "gpt-3.5-turbo-1106",
                "gpt-4o-2024-05-13",
                "gpt-4o",
            ):
                if hub == "openai":
                    from langchain_openai.chat_models import ChatOpenAI

                    model = ChatOpenAI(model=path, *args, **filtered_kwargs)
                elif hub == "azure-openai":
                    from langchain.chat_models.azure_openai import AzureChatOpenAI

                    model = AzureChatOpenAI(model=path, *args, **filtered_kwargs)

                return cls(hub, model, *args, **filtered_kwargs)
            elif hub == "ollama":
                from langchain.chat_models.ollama import ChatOllama

                model = ChatOllama(model=path, *args, **filtered_kwargs)
                return cls(hub, model, *args, **filtered_kwargs)
            else:
                model = getattr(lc, LANGCHAIN_HUBS[hub])
            default_args = inspect.getfullargspec(model).kwonlyargs
            if "model" in default_args:
                cls.model = model(model=path, *args, **filtered_kwargs)
            elif "model_name" in default_args:
                cls.model = model(model_name=path, *args, **filtered_kwargs)
            elif "model_id" in default_args:
                cls.model = model(model_id=path, *args, **filtered_kwargs)
            elif "repo_id" in default_args:
                cls.model = model(repo_id=path, model_kwargs=filtered_kwargs)
            return cls(hub, cls.model, *args, **filtered_kwargs)

        except ImportError:
            raise ValueError(Errors.E044(path=path))
        except ValidationError as e:
            error_msg = [err["loc"][0] for err in e.errors()]
            raise ConfigError(
                Errors.E045(path=path, hub=hub, field=error_msg[0], error_message=e)
            )

    @classmethod
    def _update_model_parameters(cls, hub: str, kwargs: dict):
        """Update model parameters based on the hub's mapping.

        Args:
            hub (str): The hub name for the model.
            kwargs (dict): Keyword arguments to be updated.
        """
        if hub == "azure-openai" and "deployment_name" not in kwargs:
            kwargs["deployment_name"] = "gpt-3.5-turbo-instruct"
            logging.warning(Warnings.W014(hub=hub, kwargs=kwargs))

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
            # loading a prompt manager
            from langtest.prompts import PromptManager

            prompt_manager = PromptManager()

            prompt_template = prompt_manager.get_prompt()

            if prompt_template is None:
                prompt_template = PromptTemplate(**prompt)

            llmchain = LLMChain(prompt=prompt_template, llm=self.model)
            output = llmchain.invoke(text)
            return output.get(llmchain.output_key, "")
        except Exception as e:
            raise ValueError(Errors.E089(error_message=e))

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


class PretrainedModelForNER(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for named entity recognition.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    @lru_cache(maxsize=102400)
    def predict(self, text: Union[str, dict], *args, **kwargs) -> NEROutput:
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
            prompt = {
                "input_variables": ["text"],
                "template": "Extract the named entities from the text. \n{format_instructions}\n {text}",
                "partial_variables": {
                    "format_instructions": self.__output_parser().get_format_instructions()
                },
            }
            prompt_template = PromptTemplate(**prompt)
            llmchain = LLMChain(
                prompt=prompt_template,
                llm=self.model,
                output_parser=self.__output_parser(),
            )
            result = llmchain.invoke({"text": text})
            result: dict = result.get(llmchain.output_key, {"entities": []})

            try:
                predictions = []
                for entity in result.get("entities", []):
                    try:
                        entity = NERPrediction.from_span(**entity)
                        predictions.append(entity)
                    except Exception:
                        pass

                return NEROutput(predictions=predictions)
            except Exception:
                return NEROutput(predictions=[])

        except OutputParserException:
            return NEROutput(predictions=[])

        except Exception as e:
            raise ValueError(Errors.E089(error_message=e))

    def __call__(self, text: str, *args, **kwargs):
        return self.predict(text, *args, **kwargs)

    def __output_parser(self):
        from langchain_core.output_parsers import JsonOutputParser
        from pydantic import BaseModel

        class Word(BaseModel):
            """Single word in a named entity recognition prediction"""

            word: str = Field(description="Word in the text")
            start: int = Field(
                description="Start index of the character in the word from the text"
            )
            end: int = Field(
                description="End index of the character in the word from the text"
            )
            entity: str = Field(description="Named entity type")
            score: float = Field(description="Confidence score of the prediction")
            pos_tag: str = Field(description="Part of speech tag")
            chunk_tag: str = Field(description="Chunk tag")

        class NERParser(BaseModel):
            """Named entity recognition prediction parser"""

            entities: List[Word] = Field(description="List of named entities in the text")

        parser = JsonOutputParser(pydantic_object=NERParser)
        return parser


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


class PretrainedModelForClinical(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for clinical.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForDisinformation(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for disinformation.
    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForIdeology(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for ideology.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForSensitivity(PretrainedModelForQA, ModelAPI):
    def __init__(self, hub: str, model: Any, *args, **kwargs):
        """
        Initialize the PretrainedModelForSensitivity.

        Args:
            hub (str): The hub name associated with the pretrained model.
            model (Any): The pretrained model to be used.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(hub, model, *args, **kwargs)

    @lru_cache(maxsize=102400)
    def predict(self, text: Union[str, dict], prompt, *args, **kwargs):
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
            prompt_template = PromptTemplate(**prompt)
            llmchain = LLMChain(prompt=prompt_template, llm=self.model)
            result = llmchain.run(**text)
            return {
                "result": result,
            }
        except Exception as e:
            raise ValueError(Errors.E089(error_message=e))


class PretrainedModelForWinoBias(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for wino-bias detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForLegal(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for legal.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForFactuality(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for factuality detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForSycophancy(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for sycophancy.

    This class inherits from PretrainedModelForQA and provides functionality
    specific to sycophancy task.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForVisualQA(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for visual question answering.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    @lru_cache(maxsize=102400)
    def predict(
        self, text: Union[str, dict], prompt: dict, images: List[Any], *args, **kwargs
    ):
        """Perform prediction using the pretrained model.

        Args:
            text (Union[str, dict]): The input text or dictionary.
            prompt (dict): The prompt configuration.
            images (List[Any]): The list of images.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the prediction result.
                - 'result': The prediction result.
        """
        try:
            if not isinstance(self.model, BaseChatModel):
                ValueError("visualQA task is only supported for chat models")

            # prepare prompt
            prompt_template = PromptTemplate(**prompt)
            from langchain_core.messages import HumanMessage

            images = [
                {
                    "type": "image_url",
                    "image_url": {"url": image},
                }
                for image in images
            ]

            messages = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_template.format(**text)},
                    *images,
                ]
            )

            response = self.model.invoke([messages])
            return response.content

        except Exception as e:
            raise ValueError(Errors.E089(error_message=e))
