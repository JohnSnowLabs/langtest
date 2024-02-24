from typing import Any, Callable, Union
from .modelhandler import ModelAPI
from abc import ABC
from functools import lru_cache
import importlib
from ..errors import Errors
from langtest.utils.lib_manager import try_import_lib
from ..utils.custom_types.helpers import SimplePromptTemplate
from langtest.utils.custom_types.helpers import HashableDict


def chat_completion_api(text: str, url: str, server_prompt: str, **kwargs):
    """
    Send a user text message to a chat completion API and receive the model's response.

    Args:
        text (str): The user's input text.
        url (str): The API endpoint URL.
        **kwargs: Additional keyword arguments.

    Keyword Args:
        server_prompt (str, optional): The server prompt for the chat. Defaults to a space.
        temperature (float, optional): The temperature parameter for controlling randomness. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to -1 (no limit).
        stream (bool, optional): Whether to use streaming for long conversations. Defaults to False.

    Returns:
        dict or None: The JSON response from the API if successful, otherwise None.
    """
    LIB_NAME = "requests"
    if try_import_lib(LIB_NAME):
        requests = importlib.import_module(LIB_NAME)
    else:
        raise ModuleNotFoundError(Errors.E023.format(LIB_NAME=LIB_NAME))

    if kwargs.get("headers", None):
        headers = kwargs.get("headers")
    else:
        headers = {"Content-Type": "application/json"}

    if kwargs.get("data", None):
        input_data_func = kwargs.get("data")
        data = input_data_func(text)
    else:
        server_prompt = {"role": "assistant", "content": server_prompt}
        user_text = {"role": "user", "content": text}
        data = {
            "messages": [server_prompt, user_text],
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", -1),
            "stream": kwargs.get("stream", False),
        }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(Errors.E095.format(e=str(e)))


class PretrainedModel(ABC):
    """
    Abstract base class for a custom pretrained model.

    Attributes:
        model (Any): The pretrained model to be used for prediction.

    Methods:
        load_model(cls, model: Any) -> "Any": Loads the pretrained model.
        predict(self, text: str, *args, **kwargs) -> str: Predicts the output for the given input text.
        __call__(self, text: str) -> str: Calls the predict method for the given input text.
    """

    def __init__(self, model: Any, output_parser: Callable = None, **kwargs) -> None:
        """
        Initialize the PretrainedModel.

        Args:
            model (Any): The pretrained model to be used.
            **kwargs: Additional keyword arguments.
        """
        self.model = model
        self.output_parser = output_parser
        self.kwargs = kwargs
        self.predict.cache_clear()

    @classmethod
    def load_model(cls, path: str, *args, **kwargs) -> "Any":
        """
        Load the pretrained model.

        Args:
            path (str): The path to the pretrained model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The loaded pretrained model.
        """
        if isinstance(path, dict):
            model = path["url"]
            input_data = path.get("input_processor", None)
            output_parser = path.get("output_parser", None)
            headers = path.get("headers", None)

            # missing input_processor, output_parser, headers in the dictionary
            # will raise an error
            if not all((input_data, output_parser, headers)):
                raise ValueError(
                    Errors.E090.format(
                        error_message="".join(
                            [
                                "input_processor,",
                                " output_parser",
                                " and headers",
                                " are mandatory when model is a dictionary.",
                            ]
                        )
                    )
                )
            return cls(
                model=model,
                data=input_data,
                headers=headers,
                output_parser=output_parser,
                **kwargs,
            )
        return cls(model=path, **kwargs)

    @lru_cache(maxsize=102400)
    def predict(
        self, text: Union[str, dict], prompt: dict, server_prompt, *args, **kwargs
    ):
        """
        Predicts the output for the given input text.

        Args:
            text (Union[str, dict]): The input text or dictionary.
            prompt (dict): The prompt for the prediction.
            server_prompt (str): The server prompt for the chat.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The predicted output.
        """
        try:
            prompt_template = SimplePromptTemplate(**prompt)
            p = prompt_template.format(**text)
            op = chat_completion_api(
                text=p,
                url=self.model,
                server_prompt=server_prompt,
                *args,
                **self.kwargs,
            )
            if self.output_parser:
                return self.output_parser(op)
            return op["choices"][0]["message"]["content"]
        except Exception as e:
            raise ValueError(Errors.E089.format(error_message=e))

    def predict_raw(
        self, text: Union[str, dict], prompt: dict, server_prompt: str, *args, **kwargs
    ):
        """
        Predicts the output for the given input text without caching.

        Args:
            text (Union[str, dict]): The input text or dictionary.
            prompt (dict): The prompt for the prediction.
            server_prompt (str): The server prompt for the chat.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The predicted output.
        """

        return self.predict(
            text=text, prompt=prompt, server_prompt=server_prompt, *args, **kwargs
        )

    def __call__(
        self, text: Union[str, dict], prompt: dict, server_prompt: str, *args, **kwargs
    ):
        """
        Calls the predict method for the given input text.

        Args:
            text (Union[str, dict]): The input text or dictionary.
            prompt (dict): The prompt for the prediction.
            server_prompt (str): The server prompt for the chat.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The predicted output.
        """
        if isinstance(text, dict):
            text = HashableDict(**text)
        prompt = HashableDict(**prompt)
        return self.predict(
            text=text, prompt=prompt, server_prompt=server_prompt, *args, **kwargs
        )


class PretrainedModelForQA(PretrainedModel, ModelAPI):
    """
    A class for handling a pre-trained model for question answering.

    Inherits from PretrainedModel and ModelAPI.

    Methods
    -------
    predict(text: str, *args, **kwargs)
        Predicts the answer to a given question based on the pre-trained model.

    Raises
    ------
    Exception
        If an error occurs during prediction.
    """

    pass


class PretrainedModelForSummarization(PretrainedModel, ModelAPI):
    """A class representing a pretrained model for summarization.

    Inherits:
        PretrainedModel: The base class for pretrained models.
    """

    pass


class PretrainedModelForToxicity(PretrainedModel, ModelAPI):
    """A class representing a pretrained model for toxicity detection.

    Inherits:
        PretrainedModel: The base class for pretrained models.
    """

    pass


class PretrainedModelForSecurity(PretrainedModel, ModelAPI):
    """A class representing a pretrained model for security detection.

    Inherits:
        PretrainedModel: The base class for pretrained models.
    """

    pass


class PretrainedModelForClinical(PretrainedModel, ModelAPI):
    """A class representing a pretrained model for clinical.

    Inherits:
        PretrainedModel: The base class for pretrained models.
    """

    pass


class PretrainedModelForDisinformation(PretrainedModel, ModelAPI):
    """A class representing a pretrained model for disinformation.
    Inherits:
        PretrainedModel: The base class for pretrained models.
    """

    pass


class PretrainedModelForIdeology(PretrainedModel, ModelAPI):
    """A class representing a pretrained model for ideology.

    Inherits:
        PretrainedModel: The base class for pretrained models.
    """

    pass


class PretrainedModelForSensitivity(PretrainedModel, ModelAPI):
    def __init__(self, model: Any, *args, **kwargs):
        """
        Initialize the PretrainedModelForSensitivity.

        Args:
            model (Any): The pretrained model to be used.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model, *args, **kwargs)

    @lru_cache(maxsize=102400)
    def predict(
        self, text: Union[str, dict], prompt: dict, server_prompt: str, *args, **kwargs
    ):
        """
        Perform prediction using the pretrained model.

        Args:
            text (Union[str, dict]): The input text or dictionary.
            server_prompt (str): The server prompt for the chat.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the prediction result.
                - 'result': The prediction result.
        """
        try:
            prompt_template = SimplePromptTemplate(**prompt)
            p = prompt_template.format(**text)
            op = chat_completion_api(
                text=p,
                url=self.model,
                server_prompt=server_prompt,
                *args,
                **self.kwargs,
            )
            return {
                "result": op["choices"][0]["message"]["content"],
            }

        except Exception as e:
            raise ValueError(Errors.E089.format(error_message=e))


class PretrainedModelForWinoBias(PretrainedModel, ModelAPI):
    """A class representing a pretrained model for wino-bias detection.

    Inherits:
        PretrainedModel: The base class for pretrained models.
    """

    pass


class PretrainedModelForLegal(PretrainedModel, ModelAPI):
    """A class representing a pretrained model for legal.

    Inherits:
        PretrainedModel: The base class for pretrained models.
    """

    pass


class PretrainedModelForFactuality(PretrainedModel, ModelAPI):
    """A class representing a pretrained model for factuality detection.

    Inherits:
        PretrainedModel: The base class for pretrained models.
    """

    pass


class PretrainedModelForSycophancy(PretrainedModel, ModelAPI):
    """A class representing a pretrained model for sycophancy.

    This class inherits from PretrainedModel and provides functionality
    specific to sycophancy task.

    Inherits:
        PretrainedModel: The base class for pretrained models.
    """

    pass
