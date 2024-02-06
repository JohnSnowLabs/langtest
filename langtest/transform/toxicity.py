import asyncio
from abc import ABC, abstractmethod
from typing import List

from langtest.modelhandler import ModelAPI
from ..utils.custom_types import Sample

toxicity_metric = None


class BaseToxicity(ABC):
    """Abstract base class to extend for toxicity completion"""

    alias_name = None
    supported_tasks = ["toxicity", "text-generation"]

    @staticmethod
    @abstractmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Abstract method that implements samples transformations

        Args:
            sample_list (List[Sample]): list of samples to transform

        Returns:
            List[Sample]: list of transformed samples
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(
        sample_list: List[Sample], model: ModelAPI, *args, **kwargs
    ) -> List[Sample]:
        """Computes the toxicity completion on the samples

        Args:
            sample_list (List[Sample]): list of samples to compute toxicity on
            model (ModelAPI): model to use for toxicity completion
        """
        raise NotImplementedError()

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Computes the toxicity completion on the samples in async mode.

        Args:
            sample_list (List[Sample]): list of samples to compute toxicity on
            model (ModelAPI): model to use for toxicity completion
        """
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task


class PromptToxicity(BaseToxicity):
    """Class for prompt toxicity"""

    alias_name = "offensive"

    @staticmethod
    async def run(
        sample_list: List[Sample], model: ModelAPI, *args, **kwargs
    ) -> List[Sample]:
        """Computes the toxicity completion on the samples

        Args:
            sample_list (List[Sample]): list of samples to compute toxicity on
            model (ModelAPI): model to use for toxicity completion
        """
        progress = kwargs.get("progress_bar", False)
        global toxicity_metric
        for sample in sample_list:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, *args, **kwargs)
                    if sample_status:
                        sample.completion_toxicity = toxicity_metric.compute(
                            predictions=[sample.completion]
                        )["toxicity"][0]
                        sample.state = "done"
                else:
                    sample.completion = model(sample.prompt)
                    sample.completion_toxicity = toxicity_metric.compute(
                        predictions=[sample.completion]
                    )["toxicity"][0]
                    sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list

    @staticmethod
    def transform(sample_list: List[Sample], **kwargs) -> List[Sample]:
        """Method that implements prompt toxicity transformations on the given samples

        Args:
             sample_list (List[Sample]): list of samples to compute toxicity prompt on

        Returns:
            List[Sample]:  list of transformed samples
        """
        import evaluate

        global toxicity_metric
        toxicity_metric = evaluate.load("toxicity", module_type="measurement")
        for sample in sample_list:
            sample.prompt_toxicity = toxicity_metric.compute(predictions=[sample.prompt])[
                "toxicity"
            ][0]
            sample.test_type = "offensive"
            sample.category = "toxicity"

        return sample_list


class ToxicityTypes(BaseToxicity):
    """Class for toxicity types"""

    alias_name = ["obscene", "insult", "threat", "identity_attack", "psychiatric_or_mental_illness", "homosexual_gay_or_lesbian"]

    @staticmethod
    def transform(sample_list: List[Sample], test_name) -> List[Sample]:
        """Method that implements prompt toxicity transformations on the given samples

        Args:
             sample_list (List[Sample]): list of samples to compute toxicity prompt on

        Returns:
            List[Sample]:  list of transformed samples
        """
        from transformers import pipeline

        toxicity_types = pipeline("text-classification", model="unitary/unbiased-toxic-roberta")
        for sample in sample_list:
            score = {
                x["label"]: x["score"] for x in toxicity_types(sample.prompt, top_k=None)
            }
            sample.prompt_toxicity = score[test_name]
            sample.test_type = "toxicity"
            sample.category = "toxicity"

        return sample_list

    @staticmethod
    async def run(
        sample_list: List[Sample], model: ModelAPI, *args, **kwargs
    ) -> List[Sample]:
        """Computes the toxicity completion on the samples

        Args:
            sample_list (List[Sample]): list of samples to compute toxicity on
            model (ModelAPI): model to use for toxicity completion
        """
        from transformers import pipeline

        progress = kwargs.get("progress_bar", False)

        toxicity_types = pipeline("text-classification", model="unitary/unbiased-toxic-roberta")

        for sample in sample_list:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, *args, **kwargs)
                    if sample_status:
                        sample.completion_toxicity = {
                            x["label"]: x["score"]
                            for x in toxicity_types(sample.completion, top_k=None)
                        }[sample.test_type]
                        sample.state = "done"
                else:
                    sample.completion = model(sample.prompt)
                    sample.completion_toxicity = {
                        x["label"]: x["score"]
                        for x in toxicity_types(sample.completion, top_k=None)
                    }[sample.test_type]

                    sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list
