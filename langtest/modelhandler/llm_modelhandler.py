import inspect
from typing import Any, Union
import langchain.llms as lc
from langchain import LLMChain, PromptTemplate
from pydantic import ValidationError
from ..modelhandler.modelhandler import ModelAPI, LANGCHAIN_HUBS
from ..errors import Errors

from ..metrics import EmbeddingDistance
from langchain import OpenAI
import os
from langtest.transform.utils import compare_generations_overlap


class PretrainedModelForQA(ModelAPI):
    """A class representing a pretrained model for question answering.

    Attributes:
        model: The loaded pretrained model.
        hub: The hub name for the model.
        kwargs: Additional keyword arguments.

    Raises:
        ValueError: If the model is not found online or locally.
        ConfigError: If there is an error in the model configuration.
    """

    def __init__(self, hub: str, model: Any, *args, **kwargs):
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
        if isinstance(model, str):
            self.model = self.load_model(hub, model, *args, **kwargs).model

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

            return cls(hub, cls.model, *args, **kwargs)

        except ImportError:
            raise ValueError(Errors.E044.format(path=path))
        except ValidationError as e:
            error_msg = [err["loc"][0] for err in e.errors()]
            raise ConfigError(
                Errors.E045.format(
                    path=path, hub=hub, field=error_msg[0], required_fields=error_msg
                )
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


class PretrainedModelForSummarization(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for summarization.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForToxicity(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for toxicity detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForSecurity(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for security detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForClinicalTests(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for security detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForDisinformationTest(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for disinformation test.
    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForPolitical(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for security detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForSensitivityTest(ModelAPI):
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
        if isinstance(model, str):
            model = self.load_model(model)

        from ..embeddings.openai import OpenaiEmbeddings

        self.model = model
        self.embeddings_model = OpenaiEmbeddings(model="text-embedding-ada-002")

    @classmethod
    def load_model(cls, path: str, *args, **kwargs) -> tuple:
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
            if isinstance(path, str):
                llm = OpenAI(
                    model_name=path,
                    temperature=0,
                    openai_api_key=os.environ["OPENAI_API_KEY"],
                )

                return cls(llm)
            return cls(path)
        except KeyError:
            raise ValueError(Errors.E032)

    def predict(self, text: str, text_transformed: str, test_name: str, **kwargs):
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
        expected_result_embeddings = self.embeddings_model.get_embedding(expected_result)
        actual_result_embeddings = self.embeddings_model.get_embedding(actual_result)
        if test_name == "negation":
            loss_diff = 1 - EmbeddingDistance()._cosine_distance(
                expected_result_embeddings, actual_result_embeddings
            )

        elif test_name == "toxicity":
            count1 = compare_generations_overlap(expected_result)
            count2 = compare_generations_overlap(actual_result)
            loss_diff = count2 - count1
        return {
            "loss_diff": loss_diff,
            "expected_result": expected_result,
            "actual_result": actual_result,
        }

    def __call__(self, text: str, text_transformed: str, test_name: str, **kwargs):
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

        return self.predict(
            text=text, text_transformed=text_transformed, test_name=test_name, **kwargs
        )


class PretrainedModelForWinoBias(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for wino-bias detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForLegal(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for legal-tests.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForFactualityTest(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for factuality detection.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass


class PretrainedModelForSycophancyTest(PretrainedModelForQA, ModelAPI):
    """A class representing a pretrained model for sycophancy test.

    This class inherits from PretrainedModelForQA and provides functionality
    specific to sycophancy task.

    Inherits:
        PretrainedModelForQA: The base class for pretrained models.
    """

    pass
