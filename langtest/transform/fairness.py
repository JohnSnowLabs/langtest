import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Union

from langtest.modelhandler.modelhandler import ModelAPI
from langtest.utils.custom_types import (
    MaxScoreOutput,
    MaxScoreSample,
    MinScoreOutput,
    MinScoreSample,
    Sample,
)
from langtest.utils.util_metrics import calculate_f1_score


class BaseFairness(ABC):
    """Abstract base class for implementing accuracy measures.

    Attributes:
        alias_name (str): A name or list of names that identify the accuracy measure.

    Methods:
        transform(data: List[Sample], params: Dict) -> Union[List[MinScoreSample], List[MaxScoreSample]]:
            Transforms the input data into an output based on the implemented accuracy measure.
    """

    alias_name = None
    supported_tasks = ["ner", "text-classification"]

    @staticmethod
    @abstractmethod
    def transform(
        data: List[Sample], params: Dict
    ) -> Union[List[MinScoreSample], List[MaxScoreSample]]:
        """Abstract method that implements the computation of the given measure.

        Args:
            data (List[Sample]): The input data to be transformed.
            params (Dict): parameters for tests configuration
        Returns:
            Union[List[MinScoreSample], List[MaxScoreSample]]: The transformed data based on the implemented measure.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(
        sample_list: List[MinScoreSample], categorised_data, **kwargs
    ) -> List[Sample]:
        """Computes the score for the given data.

        Args:
            sample_list (List[MinScoreSample]): The input data to be transformed.
            model (ModelAPI): The model to be used for the computation.

        Returns:
            List[MinScoreSample]: The transformed samples.
        """
        raise NotImplementedError()

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Creates a task for the run method.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for the computation.

        Returns:
            asyncio.Task: The task for the run method.

        """
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task


class MinGenderF1Score(BaseFairness):
    """Subclass of BaseFairness that implements the minimum F1 score.

    Attributes:
        alias_name (str): The name "min_gender_f1_score" identifying the minimum F1 score.

    Methods:
        transform(test: str, data: List[Sample], params: Dict) -> List[MinScoreSample]:
            Transforms the input data into an output based on the minimum F1 score.
    """

    alias_name = ["min_gender_f1_score"]

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MinScoreSample]:
        """
        Computes the minimum F1 score for the given data.

        Args:
            test (str): The test alias name.
            data (List[Sample]): The input data to be transformed.
            params (Dict): Parameters for tests configuration.

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """

        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                "male": params["min_score"],
                "female": params["min_score"],
                "unknown": params["min_score"],
            }

        samples = []
        for key, val in min_scores.items():
            sample = MinScoreSample(
                original=None,
                category="fairness",
                test_type="min_gender_f1_score",
                test_case=key,
                expected_results=MinScoreOutput(min_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MinScoreSample], grouped_label, **kwargs
    ) -> List[MinScoreSample]:
        """
        Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): The input data samples.
            grouped_label: A dictionary containing grouped labels where each key corresponds to a test case
                and the value is a tuple containing true labels and predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            List[MinScoreSample]: The evaluated data samples.
        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list:
            data = grouped_label[sample.test_case]
            if len(data[0]) > 0:
                macro_f1_score = calculate_f1_score(
                    data[0].to_list(), data[1].to_list(), average="macro", zero_division=0
                )
            else:
                macro_f1_score = 1

            sample.actual_results = MinScoreOutput(min_score=macro_f1_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MaxGenderF1Score(BaseFairness):
    """Subclass of BaseFairness that implements the maximum F1 score.

    Attributes:
        alias_name (str): The name "max_gender_f1_score" identifying the maximum F1 score.

    Methods:
        transform(test: str, data: List[Sample], params: Dict) -> List[MaxScoreSample]:
            Transforms the input data into an output based on the maximum F1 score.
    """

    alias_name = ["max_gender_f1_score"]

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MaxScoreSample]:
        """
        Computes the maximum F1 score for the given data.

        Args:
            test (str): The test alias name.
            data (List[Sample]): The input data to be transformed.
            params (Dict): Parameters for tests configuration.

        Returns:
            List[MaxScoreSample]: The transformed data based on the maximum F1 score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"
        if isinstance(params["max_score"], dict):
            max_scores = params["max_score"]
        elif isinstance(params["max_score"], float):
            max_scores = {
                "male": params["max_score"],
                "female": params["max_score"],
                "unknown": params["max_score"],
            }

        samples = []
        for key, val in max_scores.items():
            sample = MaxScoreSample(
                original=None,
                category="fairness",
                test_type="max_gender_f1_score",
                test_case=key,
                expected_results=MaxScoreOutput(max_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MaxScoreSample], grouped_label, **kwargs
    ) -> List[MaxScoreSample]:
        """
        Computes the maximum F1 score for the given data.

        Args:
            sample_list (List[MaxScoreSample]): The input data samples.
            grouped_label: A dictionary containing grouped labels where each key corresponds to a test case
                and the value is a tuple containing true labels and predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            List[MaxScoreSample]: The evaluated data samples.
        """
        progress = kwargs.get("progress_bar", False)

        for sample in sample_list:
            data = grouped_label[sample.test_case]
            if len(data[0]) > 0:
                macro_f1_score = calculate_f1_score(
                    data[0].to_list(), data[1].to_list(), average="macro", zero_division=0
                )
            else:
                macro_f1_score = 1

            sample.actual_results = MaxScoreOutput(max_score=macro_f1_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MinGenderRougeScore(BaseFairness):
    """
    Subclass of BaseFairness that implements the minimum Rouge score.

    Attributes:
        alias_name (List[str]): Alias names for the evaluation method.
        supported_tasks (List[str]): Supported tasks for this evaluation method.

    Methods:
        transform(test: str, data: List[Sample], params: Dict) -> List[MinScoreSample]:
            Transforms the input data into an output based on the minimum Rouge score.
        run(sample_list: List[MinScoreSample], grouped_label, **kwargs) -> List[MinScoreSample]:
            Computes the minimum Rouge score for the given data.

    """

    alias_name = [
        "min_gender_rouge1_score",
        "min_gender_rouge2_score",
        "min_gender_rougeL_score",
        "min_gender_rougeLsum_score",
    ]
    supported_tasks = ["question-answering", "summarization"]

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MinScoreSample]:
        """
        Transforms the data for evaluation based on the minimum Rouge score.

        Args:
            test (str): The test alias name.
            data (List[Sample]): The input data to be transformed.
            params (Dict): Parameters for tests configuration.

        Returns:
            List[MinScoreSample]: The transformed data samples based on the minimum Rouge score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                "male": params["min_score"],
                "female": params["min_score"],
                "unknown": params["min_score"],
            }

        samples = []
        for key, val in min_scores.items():
            sample = MinScoreSample(
                original=None,
                category="fairness",
                test_type=test,
                test_case=key,
                expected_results=MinScoreOutput(min_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MinScoreSample], grouped_label, **kwargs
    ) -> List[MinScoreSample]:
        """
        Computes the minimum Rouge score for the given data.

        Args:
            sample_list (List[MinScoreSample]): The input data samples.
            grouped_label: A dictionary containing grouped labels where each key corresponds to a test case
                and the value is a tuple containing true labels and predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            List[MinScoreSample]: The evaluated data samples.
        """
        import evaluate

        progress = kwargs.get("progress_bar", False)
        task = kwargs.get("task", None)

        for sample in sample_list:
            data = grouped_label[sample.test_case]
            if len(data[0]) > 0:
                if task == "question-answering" or task == "summarization":
                    em = evaluate.load("rouge")
                    macro_f1_score = em.compute(references=data[0], predictions=data[1])[
                        sample.test_type.split("_")[2]
                    ]
                else:
                    macro_f1_score = calculate_f1_score(
                        [x[0] for x in data[0]], data[1], average="macro", zero_division=0
                    )
            else:
                macro_f1_score = 1

            sample.actual_results = MinScoreOutput(min_score=macro_f1_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MaxGenderRougeScore(BaseFairness):
    """
    Subclass of BaseFairness that implements the Rouge score.

    Attributes:
        alias_name (List[str]): Alias names for the evaluation method.
        supported_tasks (List[str]): Supported tasks for this evaluation method.

    Methods:
        transform(test: str, data: List[Sample], params: Dict) -> List[MaxScoreSample]:
            Transforms the input data into an output based on the Rouge score.
        run(sample_list: List[MaxScoreSample], grouped_label, **kwargs) -> List[MaxScoreSample]:
            Computes the maximum Rouge score for the given data.

    """

    alias_name = [
        "max_gender_rouge1_score",
        "max_gender_rouge2_score",
        "max_gender_rougeL_score",
        "max_gender_rougeLsum_score",
    ]
    supported_tasks = ["question-answering", "summarization"]

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MaxScoreSample]:
        """
        Transforms the data for evaluation based on the Rouge score.

        Args:
            test (str): The test alias name.
            data (List[Sample]): The input data to be transformed.
            params (Dict): Parameters for tests configuration.

        Returns:
            List[MaxScoreSample]: The transformed data samples based on the Rouge score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        if isinstance(params["max_score"], dict):
            max_scores = params["max_score"]
        elif isinstance(params["max_score"], float):
            max_scores = {
                "male": params["max_score"],
                "female": params["max_score"],
                "unknown": params["max_score"],
            }

        samples = []
        for key, val in max_scores.items():
            sample = MaxScoreSample(
                original=None,
                category="fairness",
                test_type=test,
                test_case=key,
                expected_results=MaxScoreOutput(max_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MaxScoreSample], grouped_label, **kwargs
    ) -> List[MaxScoreSample]:
        """
        Computes the maximum Rouge score for the given data.

        Args:
            sample_list (List[MaxScoreSample]): The input data samples.
            grouped_label: A dictionary containing grouped labels where each key corresponds to a test case
                and the value is a tuple containing true labels and predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            List[MaxScoreSample]: The evaluated data samples.
        """
        import evaluate

        progress = kwargs.get("progress_bar", False)
        task = kwargs.get("task", None)

        for sample in sample_list:
            data = grouped_label[sample.test_case]
            if len(data[0]) > 0:
                if task == "question-answering" or task == "summarization":
                    em = evaluate.load("rouge")
                    rouge_score = em.compute(references=data[0], predictions=data[1])[
                        sample.test_type.split("_")[2]
                    ]
                else:
                    rouge_score = calculate_f1_score(
                        [x[0] for x in data[0]], data[1], average="macro", zero_division=0
                    )
            else:
                rouge_score = 1

            sample.actual_results = MaxScoreOutput(max_score=rouge_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MinGenderLLMEval(BaseFairness):
    """
    Class for evaluating fairness based on minimum gender performance in question-answering tasks using a Language Model.

    Attributes:
        alias_name (List[str]): Alias names for the evaluation method.
        supported_tasks (List[str]): Supported tasks for this evaluation method.

    Methods:
        transform(cls, test: str, data: List[Sample], params: Dict) -> List[MaxScoreSample]: Transforms data for evaluation.
        run(sample_list: List[MaxScoreSample], grouped_label: Dict[str, Tuple[List, List]], **kwargs) -> List[MaxScoreSample]: Runs the evaluation process.

    """

    alias_name = ["min_gender_llm_eval"]
    supported_tasks = ["question-answering"]
    eval_model = None

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MaxScoreSample]:
        """
        Transforms the data for evaluation.

        Args:
            test (str): The test alias name.
            data (List[Sample]): The data to be transformed.
            params (Dict): Parameters for transformation.

        Returns:
            List[MaxScoreSample]: The transformed data samples.

        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        from ..langtest import EVAL_MODEL
        from ..langtest import HARNESS_CONFIG as harness_config

        model = params.get("model", None)
        hub = params.get("hub", None)
        if model and hub:
            from ..tasks import TaskManager

            load_eval_model = TaskManager("question-answering")
            cls.eval_model = load_eval_model.model(
                model, hub, **harness_config.get("model_parameters", {})
            )
        else:
            cls.eval_model = EVAL_MODEL

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                "male": params["min_score"],
                "female": params["min_score"],
                "unknown": params["min_score"],
            }

        samples = []
        for key, val in min_scores.items():
            sample = MinScoreSample(
                original=None,
                category="fairness",
                test_type=test,
                test_case=key,
                expected_results=MinScoreOutput(min_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MaxScoreSample], grouped_label, **kwargs
    ) -> List[MaxScoreSample]:
        """
        Runs the evaluation process using Language Model.

        Args:
            sample_list: The input data samples.
            grouped_label: A dictionary containing grouped labels where each key corresponds to a test case
                and the value is a tuple containing true labels and predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            The evaluated data samples.
        """

        grouped_data = kwargs.get("grouped_data")
        from ..utils.custom_types.helpers import is_pass_llm_eval

        eval_model = MinGenderLLMEval.eval_model
        progress = kwargs.get("progress_bar", False)

        def eval():
            results = []
            for true_list, pred, sample in zip(data[0], data[1], X_test):
                result = is_pass_llm_eval(
                    eval_model=eval_model,
                    dataset_name=sample.dataset_name,
                    original_question=sample.original_question,
                    answer="\n".join(map(str, true_list)),
                    perturbed_question=sample.original_question,
                    prediction=pred,
                )
                if result:
                    results.append(1)
                else:
                    results.append(0)
            total_samples = len(results)
            passed_samples = sum(results)
            accuracy = passed_samples / max(total_samples, 1)
            return accuracy

        for sample in sample_list:
            data = grouped_label[sample.test_case]
            X_test = grouped_data[sample.test_case]
            if len(data[0]) > 0:
                eval_score = eval()
            else:
                eval_score = 1

            sample.actual_results = MinScoreOutput(min_score=eval_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MaxGenderLLMEval(BaseFairness):
    """
    Class for evaluating fairness based on maximum gender performance in question-answering tasks using Language Model.

    Attributes:
        alias_name (List[str]): Alias names for the evaluation method.
        supported_tasks (List[str]): Supported tasks for this evaluation method.

    Methods:
        transform(cls, test: str, data: List[Sample], params: Dict) -> List[MaxScoreSample]:
            Transforms data for evaluation.
        run(sample_list: List[MaxScoreSample], grouped_label, **kwargs) -> List[MaxScoreSample]:
            Runs the evaluation process.

    """

    alias_name = ["max_gender_llm_eval"]
    supported_tasks = ["question-answering"]
    eval_model = None

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MaxScoreSample]:
        """
        Transforms the data for evaluation.

        Args:
            test (str): The test alias name.
            data (List[Sample]): The data to be transformed.
            params (Dict): Parameters for transformation.

        Returns:
            List[MaxScoreSample]: The transformed data samples based on the maximum score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        from ..langtest import EVAL_MODEL
        from ..langtest import HARNESS_CONFIG as harness_config

        model = params.get("model", None)
        hub = params.get("hub", None)
        if model and hub:
            from ..tasks import TaskManager

            load_eval_model = TaskManager("question-answering")
            cls.eval_model = load_eval_model.model(
                model, hub, **harness_config.get("model_parameters", {})
            )
        else:
            cls.eval_model = EVAL_MODEL

        if isinstance(params["max_score"], dict):
            max_scores = params["max_score"]
        elif isinstance(params["max_score"], float):
            max_scores = {
                "male": params["max_score"],
                "female": params["max_score"],
                "unknown": params["max_score"],
            }

        samples = []
        for key, val in max_scores.items():
            sample = MaxScoreSample(
                original=None,
                category="fairness",
                test_type=test,
                test_case=key,
                expected_results=MaxScoreOutput(max_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MaxScoreSample], grouped_label, **kwargs
    ) -> List[MaxScoreSample]:
        """
        Runs the evaluation process using Language Model.

        Args:
            sample_list (List[MaxScoreSample]): The input data samples.
            grouped_label: A dictionary containing grouped labels where each key corresponds to a test case
                and the value is a tuple containing true labels and predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            List[MaxScoreSample]: The evaluated data samples.
        """
        grouped_data = kwargs.get("grouped_data")
        from ..utils.custom_types.helpers import is_pass_llm_eval

        eval_model = MaxGenderLLMEval.eval_model
        progress = kwargs.get("progress_bar", False)

        def eval():
            results = []
            for true_list, pred, sample in zip(data[0], data[1], X_test):
                result = is_pass_llm_eval(
                    eval_model=eval_model,
                    dataset_name=sample.dataset_name,
                    original_question=sample.original_question,
                    answer="\n".join(map(str, true_list)),
                    perturbed_question=sample.original_question,
                    prediction=pred,
                )
                if result:
                    results.append(1)
                else:
                    results.append(0)
            total_samples = len(results)
            passed_samples = sum(results)
            accuracy = passed_samples / max(total_samples, 1)
            return accuracy

        for sample in sample_list:
            data = grouped_label[sample.test_case]
            X_test = grouped_data[sample.test_case]
            if len(data[0]) > 0:
                eval_score = eval()
            else:
                eval_score = 1

            sample.actual_results = MaxScoreOutput(max_score=eval_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list
