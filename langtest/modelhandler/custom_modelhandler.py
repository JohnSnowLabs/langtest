from typing import Any
from langtest.utils.custom_types.output import SequenceClassificationOutput

from langtest.utils.custom_types.predictions import SequenceLabel
from .modelhandler import ModelAPI
from abc import ABC, abstractmethod
import logging
from ..errors import Errors
from functools import lru_cache

logger = logging.getLogger(__name__)


class PretrainedCustomModel(ABC):
    """
    Abstract base class for a custom pretrained model.

    Attributes:
        model (Any): The pretrained model to be used for prediction.

    Methods:
        load_model(cls, model: Any) -> "Any": Loads the pretrained model.
        predict(self, text: str, *args, **kwargs): Predicts the output for the given input text.
        __call__(self, text: str) -> None: Calls the predict method for the given input text.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        if not hasattr(self.model, "predict"):
            raise ValueError(Errors.E037)

        self.predict.cache_clear()

    @classmethod
    def load_model(cls, path: Any) -> "Any":
        return cls(path)

    @abstractmethod
    @lru_cache(maxsize=102400)
    def predict(self, text: str, *args, **kwargs):
        try:
            return self.model.predict(text, *args, **kwargs)
        except Exception as e:
            logger.error(e)
            raise e

    def predict_raw(self, text: str, *args, **kwargs):
        return self.predict(text, *args, **kwargs)

    def __call__(self, text: str) -> None:
        return self.predict(text=text)


class PretrainedModelForTextClassification(PretrainedCustomModel, ModelAPI):
    """
    A custom model handler for text classification using a pre-trained model.

    Args:
        PretrainedCustomModel: A class for loading a pre-trained custom model.
        ModelAPI: A class for handling the model.

    Methods:
        predict: Predicts the class label for a given text input.

    Returns:
        SequenceClassificationOutput: A class containing the predicted class label and score.
    """

    @lru_cache(maxsize=102400)
    def predict(self, text: str, *args, **kwargs) -> SequenceClassificationOutput:
        try:
            out = self.model.predict(text, *args, **kwargs)
            if not isinstance(out, list):
                out = [out]
            return SequenceClassificationOutput(
                predictions=[SequenceLabel(label=elt, score=1) for elt in out]
            )
        except Exception as e:
            logger.error(e)
            raise e


class PretrainedModelForQA(PretrainedCustomModel, ModelAPI):
    """
    A class for handling a pre-trained model for question answering.

    Inherits from PretrainedCustomModel and ModelAPI.

    Methods
    -------
    predict(text: str, *args, **kwargs)
        Predicts the answer to a given question based on the pre-trained model.

    Raises
    ------
    Exception
        If an error occurs during prediction.
    """

    @lru_cache(maxsize=102400)
    def predict(self, text: str, *args, **kwargs):
        try:
            out = self.model.predict(text, *args, **kwargs)
            return out
        except Exception as e:
            raise e


class PretrainedModelForCrowsPairs(PretrainedCustomModel, ModelAPI):
    """
    A custom model handler for pretrained models that are used for crows pairs.
    """

    pass


class PretrainedModelForWinoBias(PretrainedCustomModel, ModelAPI):
    """
    A custom model handler for pretrained models that are used for wino bias.
    """

    pass


class PretrainedModelForTranslation(PretrainedCustomModel, ModelAPI):
    """
    A custom model handler for pretrained models that are used for translation.
    """

    pass


class PretrainedModelForSummarization(PretrainedCustomModel, ModelAPI):
    """
    A custom model handler for pretrained models that are used for summarization.
    """

    pass


class PretrainedModelForToxicity(PretrainedCustomModel, ModelAPI):
    """
    A custom model handler for pretrained models that are used for toxicity.
    """

    pass


class PretrainedModelForSecurity(PretrainedCustomModel, ModelAPI):
    """
    A custom model handler for pretrained models that are used for security.
    """

    pass


class PretrainedModelForPolitical(PretrainedCustomModel, ModelAPI):
    """
    A custom model handler for pretrained models that are used for political.
    """

    pass


class PretrainedModelForDisinformationTest(PretrainedCustomModel, ModelAPI):
    """
    A custom model handler for pretrained models that are used for disinformation.
    """

    pass


class PretrainedModelForFactualityTest(PretrainedCustomModel, ModelAPI):
    """
    A class representing a pretrained model for factuality test.

    Inherits from PretrainedCustomModel and ModelAPI.
    """

    pass


class PretrainedModelForSensitivityTest(PretrainedCustomModel, ModelAPI):
    """
    A class representing a pre-trained model for sensitivity testing.
    Inherits from PretrainedCustomModel and ModelAPI.
    """

    pass


class PretrainedModelForSycophancyTest(PretrainedCustomModel, ModelAPI):
    """
    A custom model handler for testing sycophancy using a pre-trained model.
    Inherits from PretrainedCustomModel and ModelAPI.
    """

    pass
