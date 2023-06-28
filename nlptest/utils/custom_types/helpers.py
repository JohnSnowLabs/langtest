from pydantic import BaseModel
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class Span(BaseModel):
    """Representation of a text's slice"""
    start: int
    end: int
    word: str

    @property
    def ends_with_space(self) -> bool:
        """"""
        return self.word.endswith(" ")

    def shift_start(self, offset: int) -> None:
        """"""
        self.start -= offset

    def shift_end(self, offset: int) -> None:
        """"""
        self.end -= offset

    def shift(self, offset: int) -> None:
        """"""
        self.start -= offset
        self.end -= offset

    def __hash__(self):
        """"""
        return hash(self.__repr__())

    def __eq__(self, other):
        """"""
        return self.start == other.start and \
               self.end - int(self.ends_with_space) == other.end - int(other.ends_with_space)

    def __str__(self):
        """"""
        return f"<Span(start={self.start}, end={self.end}, word='{self.word}')>"

    def __repr__(self):
        """"""
        return f"<Span(start={self.start}, end={self.end}, word='{self.word}')>"


class Transformation(BaseModel):
    """
    Helper object keeping track of an alteration performed on a piece of text.
    It holds information about how a given span was transformed into another one
    """
    original_span: Span
    new_span: Span
    ignore: bool = False

class SimpleSentenceTransformer:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences, convert_to_tensor=False, max_length=128):
        # Ensure sentences is a list
        if not isinstance(sentences, list):
            sentences = [sentences]

        # Tokenize the sentences
        encoded_input = self.tokenizer(sentences, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(self.device)

        # Get the model's output
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling.
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        if convert_to_tensor:
            return sentence_embeddings
        else:
            return sentence_embeddings.cpu().numpy()
        
def cosine_similarity(array1, array2):
 
    dot_products = np.einsum('ij,ij->i', array1, array2)
    magnitudes1 = np.linalg.norm(array1, axis=1)
    magnitudes2 = np.linalg.norm(array2, axis=1)
    cosine_similarities = dot_products / (magnitudes1 * magnitudes2)

    return cosine_similarities