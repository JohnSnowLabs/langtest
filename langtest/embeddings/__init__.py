from .huggingface import HuggingfaceEmbeddings
from .openai import OpenAIEmbeddings

embedding_info = {
    "openai": {"class": OpenAIEmbeddings, "default_model": "text-embedding-ada-002"},
    "huggingface": {
        "class": HuggingfaceEmbeddings,
        "default_model": "sentence-transformers/all-mpnet-base-v2",
    },
}
