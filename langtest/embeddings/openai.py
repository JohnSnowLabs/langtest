import os
import importlib
from typing import Union
import numpy as np
from ..utils.lib_manager import try_import_lib
from tenacity import retry, wait_random_exponential, stop_after_attempt
from ..errors import Errors


class OpenaiEmbeddings:
    LIB_NAME = "openai"

    def __init__(self, model="text-embedding-ada-002"):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.openai = None
        self._check_openai_package()
        if not self.api_key:
            raise ValueError(Errors.E032())

        self.openai.api_key = self.api_key

    def _check_openai_package(self):
        """Check if the 'openai' package is installed and import the required functions.

        Raises an error if the package is not found.
        """
        if try_import_lib(self.LIB_NAME):
            self.openai = importlib.import_module(self.LIB_NAME)
        else:
            raise ModuleNotFoundError(Errors.E023(LIB_NAME=self.LIB_NAME))

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(
        self, text: Union[str, list], convert_to_tensor: bool = False
    ) -> Union[np.ndarray, list]:
        """
        Get an embedding for the input text using OpenAI's text-embedding models.

        Args:
            text (str): The input text to embed.

        Returns:
            list[float]: A list of floating-point values representing the text's embedding.
        """
        if isinstance(text, list):
            response = self.openai.Embedding.create(input=text, model=self.model)
            embedding = [
                np.array(response["data"][i]["embedding"]).reshape(1, -1)
                for i in range(len(text))
            ]
            return embedding
        else:
            response = self.openai.Embedding.create(input=[text], model=self.model)
            embedding = np.array(response["data"][0]["embedding"]).reshape(1, -1)
            return embedding
