import os
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt


class OpenAIEmbeddings:
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
            )

        openai.api_key = self.api_key

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(self, text: str) -> list[float]:
        """
        Get an embedding for the input text using OpenAI's text-embedding models.

        Args:
            text (str): The input text to embed.

        Returns:
            list[float]: A list of floating-point values representing the text's embedding.
        """
        response = openai.Embedding.create(input=[text], model=self.model)
        embedding = response["data"][0]["embedding"]
        return embedding
