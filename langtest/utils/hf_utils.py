import logging
from typing import Any, List, Optional
import importlib.util
from ..errors import Errors, Warnings
from langtest.utils.lib_manager import try_import_lib
import os
import random
from typing import Tuple, Dict


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
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )
        from huggingface_hub import login
    except ImportError:
        raise ValueError(Errors.E085)

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


class HuggingFacePipeline:
    def __init__(
        self,
        model_id: Optional[str] = None,
        task: Optional[str] = None,
        device: Optional[int] = -1,
        pipeline: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.model_id = model_id
        if pipeline:
            self.pipeline = pipeline
        else:
            self.pipeline = self._initialize_pipeline(model_id, task, device, **kwargs)

    def _initialize_pipeline(
        self, model_id: str, task: str, device: Optional[int], **kwargs: Any
    ) -> Any:
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ValueError(Errors.E085)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Set the pad_token_id for the tokenizer
        tokenizer.pad_token_id = tokenizer.eos_token_id

        try:
            if task == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(model_id)
            elif task in ("text2text-generation", "summarization"):
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            else:
                raise ValueError(Errors.E086.format(task=task))
        except ImportError as e:
            raise ValueError(Errors.E087.format(task=task)) from e

        if (
            getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)
        ) and device is not None:
            logging.warning(Warnings.W015.format(device=device))
            device = None

        if device is not None and importlib.util.find_spec("torch") is not None:
            import torch

            cuda_device_count = torch.cuda.device_count()
            if device < -1 or (device >= cuda_device_count):
                raise ValueError(
                    Errors.E088.format(device=device, cuda_device_count=cuda_device_count)
                )
            if device < 0 and cuda_device_count > 0:
                logging.warning(Warnings.W016.format(cuda_device_count=cuda_device_count))

        return hf_pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            device=device,
            **kwargs,
        )

    def _generate(self, prompts: List[str]) -> List[str]:
        text_generations: List[str] = []

        for prompt in prompts:
            response = self.pipeline(prompt)

            if isinstance(response, list):
                response = response[0]

            if self.pipeline.task == "text-generation":
                try:
                    from transformers.pipelines.text_generation import ReturnType

                    remove_prompt = (
                        self.pipeline._postprocess_params.get("return_type")
                        != ReturnType.NEW_TEXT
                    )
                except Exception as e:
                    logging.warning(Warnings.W017.format(e=e))
                    remove_prompt = True
                if remove_prompt:
                    text = response["generated_text"][len(prompt) :]
                else:
                    text = response["generated_text"]
            elif self.pipeline.task == "text2text-generation":
                text = response["generated_text"]
            elif self.pipeline.task == "summarization":
                text = response["summary_text"]
            else:
                raise ValueError(Errors.E086.format(task=self.pipeline.task))

            text_generations.append(text)

        return text_generations
