from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import os
from huggingface_hub import login


class GatedRepoAccessError(Exception):
    """
    Exception raised when there is an attempt to access a gated Hugging Face repository without proper authorization.
    """

    pass


def get_model_n_tokenizer(model_name):
    """
    Load a pre-trained model and tokenizer from Hugging Face Model Hub.

    Args:
        model_name (str): The name or identifier of the pre-trained model.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
            - model: The loaded pre-trained model.
            - tokenizer: The tokenizer associated with the model.

    Raises:
        GatedRepoAccessError: If there is an attempt to access a gated repository without proper authorization.
    """
    if "HUGGINGFACEHUB_API_TOKEN" in os.environ:
        login(os.environ["HUGGINGFACEHUB_API_TOKEN"])

    try:
        # Try loading the model as AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name)

    except ValueError:
        # Try loading the model as AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    except OSError:
        raise GatedRepoAccessError(
            "You are trying to access a gated repo. "
            "Make sure to request access at "
            f"{model_name} and pass a token having permission to this repo either by logging in with `huggingface-cli login` or by setting the `HUGGINGFACEHUB_API_TOKEN` environment variable with your API token."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer
