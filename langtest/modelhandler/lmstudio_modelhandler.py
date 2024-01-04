from typing import Any
from .modelhandler import ModelAPI
from abc import ABC
import logging
import requests
from functools import lru_cache

logger = logging.getLogger(__name__)
from ..utils.custom_types.helpers import SimplePromptTemplate


def chat_completion_api(text, url, server_prompt, **kwargs):
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
    headers = {"Content-Type": "application/json"}
    server_prompt = {"role": "assistant", "content": server_prompt}
    user_text = {"role": "user", "content": text}

    data = {
        "messages": [server_prompt, user_text],
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", -1),
        "stream": kwargs.get("stream", False),
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


class PretrainedModel(ABC):
    """
    Abstract base class for a custom pretrained model.

    Attributes:
        model (Any): The pretrained model to be used for prediction.

    Methods:
        load_model(cls, model: Any) -> "Any": Loads the pretrained model.
        predict(self, text: str, *args, **kwargs): Predicts the output for the given input text.
        __call__(self, text: str) -> None: Calls the predict method for the given input text.
    """

    def __init__(self, model: Any, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs
        self.predict.cache_clear()

    @classmethod
    def load_model(cls, path: Any, **kwargs) -> "Any":
        return cls(path, **kwargs)

    @lru_cache(maxsize=1024000)
    def predict(self, text: str, server_prompt, *args, **kwargs):
        try:
            op = chat_completion_api(
                text=text,
                url=self.model,
                server_prompt=server_prompt,
                *args,
                **self.kwargs,
            )
            return op["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(e)
            raise e

    def predict_raw(self, text: str, prompt, server_prompt, *args, **kwargs):
        prompt_template = SimplePromptTemplate(**prompt)
        p = prompt_template.format(**text)
        return self.predict(text=p, server_prompt=server_prompt, *args, **kwargs)

    def __call__(self, text: str, prompt, server_prompt, *args, **kwargs) -> None:
        prompt_template = SimplePromptTemplate(**prompt)
        p = prompt_template.format(**text)
        return self.predict(text=p, server_prompt=server_prompt, *args, **kwargs)


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
