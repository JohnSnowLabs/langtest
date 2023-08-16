from typing import Tuple, List, Union
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


class SimpleSentenceTransformer:
    """A simple class to handle the sentence transformation using the specified model.

    Attributes:
        device (torch.device): The device used for computations, i.e., either a GPU (if available) or a CPU.
        tokenizer (transformers.AutoTokenizer): The tokenizer associated with the model.
        model (transformers.AutoModel): The transformer model used for sentence embeddings.
    """

    def __init__(
        self,
        model_name: str,
    ):
        """Constructor method

        Args:
            model_name (str): The name of the model to be loaded. By default, it uses the multilingual MiniLM model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def mean_pooling(
        self, model_output: Tuple[torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean pooling on the model outputs.

        Args:
            model_output (torch.Tensor): The model's output.
            attention_mask (torch.Tensor): The attention mask tensor.

        Returns:
            torch.Tensor: The mean pooled output tensor.
        """
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_tensor: bool = False,
        max_length: int = 128,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Encode sentences into sentence embeddings.

        Args:
            sentences (str | list): The sentences to be encoded. Can be either a single string or a list of strings.
            convert_to_tensor (bool, optional): If set to True, the method will return tensors, otherwise it will return numpy arrays. Defaults to False.
            max_length (int, optional): The maximum length for the sentences. Any sentence exceeding this length gets truncated. Defaults to 128.

        Returns:
            torch.Tensor | numpy.ndarray: The sentence embeddings. The datatype depends on the 'convert_to_tensor' parameter.
        """
        # Ensure sentences is a list
        if not isinstance(sentences, list):
            sentences = [sentences]

        # Tokenize the sentences
        encoded_input = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        # Get the model's output
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling.
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        if convert_to_tensor:
            return sentence_embeddings
        else:
            return sentence_embeddings.cpu().numpy()
