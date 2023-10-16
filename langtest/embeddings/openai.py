import os
import importlib
import numpy as np
from ..utils.lib_manager import try_import_lib
from tenacity import retry, wait_random_exponential, stop_after_attempt


class OpenAIEmbeddings:
    LIB_NAME = "openai"

    def __init__(self, model="text-embedding-ada-002"):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self._check_openai_package()
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
            )

        self.openai.api_key = self.api_key

    def _check_openai_package(self):
        """Check if the 'openai' package is installed and import the required functions.

        Raises an error if the package is not found.
        """
        if try_import_lib(self.LIB_NAME):
            self.openai = importlib.import_module(self.LIB_NAME)
        else:
            raise ModuleNotFoundError(
                f"The '{self.LIB_NAME}' package is not installed. Please install it using 'pip install {self.LIB_NAME}'."
            )

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(self, text: str) -> list[float]:
        """
        Get an embedding for the input text using OpenAI's text-embedding models.

        Args:
            text (str): The input text to embed.

        Returns:
            list[float]: A list of floating-point values representing the text's embedding.
        """
        response = self.openai.Embedding.create(input=[text], model=self.model)
        embedding = np.array(response["data"][0]["embedding"]).reshape(1, -1)
        return embedding
