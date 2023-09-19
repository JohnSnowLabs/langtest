import inspect
from typing import Union
import langchain.llms as lc
from langchain import LLMChain, PromptTemplate
from pydantic import ValidationError
from ..modelhandler.modelhandler import _ModelHandler, LANGCHAIN_HUBS

from ..utils.util_metrics import cosine_similarity
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import os


class PretrainedModelForQA(_ModelHandler):
    """A class representing a pretrained model for question answering.

    Attributes:
        model: The loaded pretrained model.
        hub: The hub name for the model.
        kwargs: Additional keyword arguments.

    Raises:
        ValueError: If the model is not found online or locally.
        ConfigError: If there is an error in the model configuration.
    """

    def __init__(self, hub: str, model: str, *args, **kwargs):
        """Constructor class

        Args:
            hub (str): The hub name for the model.
            model (str): The model name or path.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.model = model
        self.hub = LANGCHAIN_HUBS[hub]
        self.kwargs = kwargs

    @classmethod
    def load_model(cls, hub: str, path: str, *args, **kwargs) -> "PretrainedModelForQA":
        """Load the pretrained model.

        Args:
            hub (str): The hub name for the model.
            path (str): The model path.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            PretrainedModelForQA: The loaded pretrained model.

        Raises:
            ValueError: If the model is not found online or locally.
            ConfigError: If there is an error in the model configuration.
        """
        try:
            model = getattr(lc, LANGCHAIN_HUBS[hub])
            default_args = inspect.getfullargspec(model).kwonlyargs
            if "model" in default_args:
                cls.model = model(model=path, *args, **kwargs)
            elif "model_name" in default_args:
                cls.model = model(model_name=path, *args, **kwargs)
            elif "model_id" in default_args:
                cls.model = model(model_id=path, *args, **kwargs)
            elif "repo_id" in default_args:
                cls.model = model(repo_id=path, model_kwargs=kwargs)
            return cls.model
        except ImportError:
            raise ValueError(
                f"""Model "{path}" is not found online or local.
                Please install langchain by pip install langchain"""
            )
        except ValidationError as e:
            error_msg = [err["loc"][0] for err in e.errors()]
            raise ConfigError(
                f"\nPlease update model_parameters section in config.yml file for {path} model in {hub}.\nmodel_parameters:\n\t{error_msg[0]}: value \n\n{error_msg} is required field(s), please provide them in config.yml "
            )

    def predict(self, text: Union[str, dict], prompt: dict, *args, **kwargs):
        """Perform prediction using the pretrained model.

        Args:
            text (Union[str, dict]): The input text or dictionary.
            prompt (dict): The prompt configuration.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The prediction result.
        """
        prompt_template = PromptTemplate(**prompt)
        llmchain = LLMChain(prompt=prompt_template, llm=self.model)
        return llmchain.run(**text)

    def predict_raw(self, text: Union[str, dict], prompt: dict, *args, **kwargs):
        """Perform raw prediction using the pretrained model.

        Args:
            text (Union[str, dict]): The input text or dictionary.
            prompt (dict): The prompt configuration.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The prediction result.
        """
        return self.predict(text, prompt, *args, **kwargs)

    def __call__(self, text: Union[str, dict], prompt: dict, *args, **kwargs):
        """Perform prediction using the model.

        Args:
            text (Union[str, dict]): The input text or dictionary.
            prompt (dict): The prompt configuration.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The prediction result.
        """
        return self.predict(text, prompt, *args, **kwargs)


class ConfigError(BaseException):
    """An exception raised for configuration errors.

    Args:
        message (str): The error message.

    Attributes:
        message (str): The error message.

    Examples:
        >>> raise ConfigError('Invalid configuration')
    """

    def __init__(self, message: str):
        """Constructor method

        Args:
             message (str): message to display
        """
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class PretrainedModelForSummarization(PretrainedModelForQA, _ModelHandler):
    """A class representing a pretrained model for summarization.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForToxicity(PretrainedModelForQA, _ModelHandler):
    """A class representing a pretrained model for toxicity detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForSecurity(PretrainedModelForQA, _ModelHandler):
    """A class representing a pretrained model for security detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForClinicalTests(PretrainedModelForQA, _ModelHandler):
    """A class representing a pretrained model for security detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForDisinformationTest(PretrainedModelForQA, _ModelHandler):
    """A class representing a pretrained model for disinformation test.
    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForPolitical(PretrainedModelForQA, _ModelHandler):
    """A class representing a pretrained model for security detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForSensitivityTest(_ModelHandler):
    """
    A class for sensitivity testing using a pretrained model and embeddings.

    Args:
        model (tuple): A tuple containing the pretrained language model and embeddings model.

    Raises:
        ValueError: If the input 'model' is not a tuple.

    Attributes:
        model: The pretrained language model.
        embeddings_model: The embeddings model.

    Methods:
        load_model(cls, path):
            Load the pretrained language model and embeddings model from a given path.
        predict(self, text, text_transformed, **kwargs):
            Predict the sensitivity of the model to text transformations.
    """

    def __init__(self, model: str):
        """
        Initialize the PretrainedModelForSensitivityTest.

        Args:
            model (tuple): A tuple containing the pretrained language model and embeddings model.

        Raises:
            ValueError: If the input 'model' is not a tuple.
        """

        self.model, self.embeddings_model = model

    @classmethod
    def load_model(cls, path: str) -> tuple:
        """
        Load the pretrained language model and embeddings model from a given path.

        Args:
            path (str): The path to the model files.

        Returns:
            tuple: A tuple containing the pretrained language model and embeddings model.

        Raises:
            ValueError: If the 'OPENAI_API_KEY' environment variable is not set.
        """
        try:
            llm = OpenAI(
                model_name=path,
                temperature=0,
                openai_api_key=os.environ["OPENAI_API_KEY"],
            )
            embeddings_model = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=os.environ["OPENAI_API_KEY"],
            )
            return llm, embeddings_model
        except KeyError:
            raise ValueError("The 'OPENAI_API_KEY' environment variable is not set.")

    def predict(self, text: str, text_transformed: str, **kwargs):
        """
        Predict the sensitivity of the model to text transformations.

        Args:
            text (str): The original text.
            text_transformed (str): The transformed text.

        Returns:
            dict: A dictionary containing the loss difference, expected result, and actual result.
                - 'loss_diff' (float): The cosine similarity-based loss difference.
                - 'expected_result' (str): The model's output for the original text.
                - 'actual_result' (str): The model's output for the transformed text.
        """

        expected_result = self.model(text)
        actual_result = self.model(text_transformed)

        expected_result_embeddings = self.embeddings_model.embed_documents(
            [expected_result]
        )
        actual_result_embeddings = self.embeddings_model.embed_documents([actual_result])

        loss = 1 - cosine_similarity(expected_result_embeddings, actual_result_embeddings)

        return {
            "loss_diff": loss[0],
            "expected_result": expected_result,
            "actual_result": actual_result,
        }

    def __call__(self, text: str, text_transformed: str, **kwargs):
        """
        Alias of the 'predict' method.

        Args:
            text (str): The original text.
            text_transformed (str): The transformed text.

        Returns:
            dict: A dictionary containing the loss difference, expected result, and actual result.
                - 'loss_diff' (float): The cosine similarity-based loss difference.
                - 'expected_result' (str): The model's output for the original text.
                - 'actual_result' (str): The model's output for the transformed text.
        """

        return self.predict(text=text, text_transformed=text_transformed, **kwargs)


class PretrainedModelForWinoBias(PretrainedModelForQA, _ModelHandler):
    """A class representing a pretrained model for wino-bias detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForLegal(PretrainedModelForQA, _ModelHandler):
    """A class representing a pretrained model for legal-tests.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForFactualityTest(PretrainedModelForQA, _ModelHandler):
    """A class representing a pretrained model for factuality detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass
