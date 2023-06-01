import evaluate
from ..utils.custom_types import Sample
from typing import List
toxicity_metric = evaluate.load("toxicity",module_type="measurement")

class BaseToxicity:

    def transform(self):
        pass

    async def async_run(self):
        pass

    async def run(self):
        pass


class PromptToxicity(BaseToxicity):
    alias_name = "offensive"

    def __init__(self, text):
        self.text = text

    def transform(sample_list: List[Sample]) -> List[Sample]:
        for sample in sample_list:
            sample.prompt_toxicity = toxicity_metric.compute(predictions=[sample.prompt])
        
        sample.test_type = "offensive"
        sample.category = "toxicity"
        
        return sample_list

    async def async_run(self):
        
        return self.transform()

    def run(self):
        return self.transform()