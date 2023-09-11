from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import login

class GatedRepoAccessError(Exception):
    pass

def get_model_n_tokenizer(model_name):
    if "HUGGINGFACEHUB_API_TOKEN" in os.environ:
        login(os.environ["HUGGINGFACEHUB_API_TOKEN"])
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer
    except OSError as e:
        error_message = str(e)
        if "You are trying to access a gated repo" in error_message:
            raise GatedRepoAccessError(
                "You are trying to access a gated repo. "
                "Make sure to request access at "
                f"{model_name} and pass a token having permission to this repo either by logging in with `huggingface-cli login` or by setting the `HUGGINGFACEHUB_API_TOKEN` environment variable with your API token."