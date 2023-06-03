import evaluate
from abc import ABC, abstractmethod
from ..utils.custom_types import Sample
from typing import List


class BaseToxicity(ABC):
    alias_name = None
    supported_tasks = ["toxicity"]

    @staticmethod
    @abstractmethod
    def transform(self):
        return NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(self):
        return NotImplementedError()

    @classmethod
    async def async_run(cls):
        return NotImplementedError()



class PromptToxicity(BaseToxicity):
    alias_name = "offensive"
    

    def transform(sample_list: List[Sample]) -> List[Sample]:
        toxicity_metric = evaluate.load("toxicity",module_type="measurement")
        for sample in sample_list:
            sample.prompt_toxicity = toxicity_metric.compute(predictions=[sample.prompt])['toxicity'][0]
            sample.test_type = "offensive"
            sample.category = "toxicity"
        
        return sample_list

    async def async_run(self, sample_list: List[Sample], model, *args, **kwargs):

        return self.transform()

    def run(self):
        return self.transform()