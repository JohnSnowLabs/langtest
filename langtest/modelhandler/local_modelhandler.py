from typing import Any
from .modelhandler import ModelAPI
from abc import ABC
import logging
import requests

logger = logging.getLogger(__name__)
from ..utils.custom_types.helpers import SimplePromptTemplate


def chat_completion_api(text, url, **kwargs):
    headers = {"Content-Type": "application/json"}

    server_prompt = {"role": "assistant", "content": kwargs.get("server_prompt", "")}
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

    @classmethod
    def load_model(cls, path: Any, **kwargs) -> "Any":
        return cls(path, **kwargs)

    def predict(self, text: str, prompt, *args, **kwargs):
        try:
            prompt_template = SimplePromptTemplate(**prompt)
            p = prompt_template.format(**text)
            op = chat_completion_api(
                text=p, url=self.model, prompt=prompt, *args, **self.kwargs
            )
            return op["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(e)
            raise e

    def predict_raw(self, text: str, prompt, *args, **kwargs):
        return self.predict(text, prompt, *args, **kwargs)

    def __call__(self, text: str, prompt) -> None:
        return self.predict(
            prompt=prompt,
            text=text,
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
