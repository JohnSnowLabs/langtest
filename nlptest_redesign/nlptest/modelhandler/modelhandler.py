from abc import ABC, abstractmethod

from transformers import pipeline

from .custom_types import HfNEROutput


class _ModelHandler(ABC):

    @abstractmethod
    def load_model(self):
        """"""
        return NotImplementedError()

    @abstractmethod
    def predict(self, text: str, **kwargs):
        """"""
        return NotImplementedError()


class ModelFactory:
    """"""

    def __init__(
            self,
            model_path: str,
            task: str
    ) -> None:
        self.model_path = model_path
        self.task = task

        self._class_map = {cls.__name__.replace("PretrainedModel", "").lower(): cls for cls in
                           _ModelHandler.__subclasses__()}
        self._load()

    def _load(self):
        """"""
        self._class_map[self.task].load_model()


class NERHFPretrainedModel(_ModelHandler):
    """"""

    def __init__(
            self,
            model_path: str
    ):
        self.model_path = model_path

    def load_model(self):
        """"""
        model = pipeline(model=self.model_path)
        assert model.task == "token-classifier"
        self.model = model

    def predict(self, text: str, **kwargs) -> HfNEROutput:
        """"""
        prediction = self.model(text)

        # depending on how we want the output to be we might want to somehow aggregate the predictions
        # example here:
        # if kwargs["group_entities"]:
        #     prediction = self.model.group_entities(prediction)
        return prediction

    def __call__(self, text: str, *args, **kwargs):
        """"""
        return self.predict(text, **kwargs)
