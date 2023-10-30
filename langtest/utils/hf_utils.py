from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import os
from ..errors import Errors
from huggingface_hub import login
import random
from typing import List, Tuple, Dict
import importlib
from langtest.utils.lib_manager import try_import_lib


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
        Tuple: A tuple containing the loaded model and tokenizer.
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
        raise GatedRepoAccessError(Errors.E071.format(model_name=model_name))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def clean_input(example: str) -> str:
    """
    Clean and format input example.

    Args:
        example (str): The input example to be cleaned.

    Returns:
        str: The cleaned and formatted input example.
    """
    example = example.replace('"', "")
    example = example.replace("\n", "")
    example = example.strip()
    example = f'"{example}"'
    return example


def build_dataset(
    dataset_name: str,
    dataset_subset: str,
    label_name: str,
    text_fields: List[str],
    natural_language_labels: List[str],
) -> Tuple[Dict[str, str], Dict[str, str], bool]:
    """
    Uses inputted dataset details to build dictionaries of train/test values.

    Args:
        dataset_name (str): The name of the dataset.
        dataset_subset (str): The name of the dataset subset.
        label_name (str): The name of the label.
        text_fields (List[str]): The list of text fields.
        natural_language_labels (List[str]): The list of natural language labels.

    Returns:
        Tuple[Dict[str, str], Dict[str, str], bool]: A tuple containing train and test dictionaries and a boolean indicating the presence of validation data.
    """

    LIB_NAME = "datasets"
    if try_import_lib(LIB_NAME):
        dataset_module = importlib.import_module(LIB_NAME)
        load_dataset = getattr(dataset_module, "load_dataset")
    else:
        raise ModuleNotFoundError(Errors.E023.format(LIB_NAME=LIB_NAME))

    """Uses inputted dataset details to build dictionary of train/test values."""
    if not dataset_subset:
        dataset_dict = load_dataset(dataset_name)
    else:
        dataset_dict = load_dataset(dataset_name, name=dataset_subset)

    train_dict, test_dict = {}, {}
    has_validation = False

    if "validation" in dataset_dict:
        train_data, test_data = dataset_dict["train"], dataset_dict["validation"]
        has_validation = True
    elif "test" in dataset_dict:
        train_data, test_data = dataset_dict["train"], dataset_dict["test"]
    else:
        temp = list(dataset_dict["train"])
        random.shuffle(temp)
        num_test_examples = int(0.2 * len(temp))
        train_data, test_data = temp[:-num_test_examples], temp[-num_test_examples:]

    for data, dict_ in zip([train_data, test_data], [train_dict, test_dict]):
        for i in range(len(data)):
            text = " and ".join([clean_input(data[i][x]) for x in text_fields])
            label = data[i][label_name]
            dict_[text] = natural_language_labels[label]

    return train_dict, test_dict, has_validation
