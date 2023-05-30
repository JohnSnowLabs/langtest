from abc import ABC, abstractmethod
import asyncio
from typing import List
from nlptest.modelhandler.modelhandler import ModelFactory
from nlptest.utils.custom_types import Sample, MinScoreOutput, MinScoreSample, MinScoreQASample
from nlptest.utils.custom_types.output import NEROutput, SequenceClassificationOutput
from nlptest.utils.gender_classifier import GenderClassifier
from .utils import (default_label_representation,
                    default_ehtnicity_representation,
                    default_economic_country_representation,
                    default_religion_representation,
                    get_label_representation_dict,
                    get_country_economic_representation_dict,
                    get_religion_name_representation_dict,
                    get_ethnicity_representation_dict,
                    get_entity_representation_proportions)


class BaseRepresentation(ABC):

    """
    Abstract base class for implementing representation measures.

    Attributes:
        alias_name (str): A name or list of names that identify the representation measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output 
        based on the implemented representation measure.
    """
    alias_name = None
    supported_tasks = ["ner", "text-classification","question-answering","summarization"]

    @staticmethod
    @abstractmethod
    def transform(self):
        """
        Abstract method that implements the representation measure.

        Args:
            data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
            Any: The transformed data based on the implemented representation measure.
        """

        return NotImplementedError

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[Sample], model: ModelFactory, **kwargs) -> List[Sample]:
        return NotImplementedError()

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelFactory, **kwargs):
        """
        Creates a task for the run method.

        Args:
            sample_list (List[Sample]): The input data to be evaluated for representation test.
            model (ModelFactory): The model to be used for the computation.

        Returns:
            asyncio.Task: The task for the run method.
        """
        created_task = asyncio.create_task(
            cls.run(sample_list, model, **kwargs))
        return created_task


class GenderRepresentation(BaseRepresentation):
    """
    Subclass of BaseRepresentation that implements the gender representation test.

    Attributes:
        alias_name (List[str]): The list of test names that identify the representation measure.

    """
    alias_name = [
        "min_gender_representation_count",
        "min_gender_representation_proportion"
    ]
    
    supported_tasks = ["ner", "text-classification","question-answering","summarization"]

    def transform(test, data, params):
        """
        Args:
            test (str): name of the test
            data (List[Sample]): The input data to be evaluated for representation test.
            params : parameters specified in config.

        Raises:
            ValueError: If sum of specified proportions in config is greater than 1

        Returns:
            List[Sample]: Gender Representation test results.
        """

        samples = []
        if test == "min_gender_representation_count":
            if isinstance(params["min_count"], dict):
                min_counts = params["min_count"]
            else:
                min_counts = {
                    "male": params["min_count"],
                    "female": params["min_count"],
                    "unknown": params["min_count"]
                }

            for key, value in min_counts.items():
                
                if hasattr(data[0], 'task') and data[0].task == 'question-answering':
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_gender_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    samples.append(sample)
                    
                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_gender_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    samples.append(sample)
        elif test == "min_gender_representation_proportion":
            min_proportions = {
                "male": 0.26,
                "female": 0.26,
                "unknown": 0.26
            }

            if isinstance(params["min_proportion"], dict):
                min_proportions = params["min_proportion"]
                if sum(min_proportions.values()) > 1:
                    raise ValueError(
                        '''Sum of proportions cannot be greater than 1. 
                        So min_gender_representation_proportion test cannot run.''')

            for key, value in min_proportions.items():
                if hasattr(data[0], 'task') and data[0].task == 'question-answering':
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_gender_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    samples.append(sample)
                    
                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_gender_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    samples.append(sample)
        return samples

    async def run(sample_list: List[Sample], model: ModelFactory, **kwargs) -> List[Sample]:

        """
        Computes the actual results for the Gender Representation test.

        Args:
            sample_list (List[Sample]): The input data to be evaluated for representation test.
            model (ModelFactory): The model factory object.

        Returns:
            List[Sample]: The list of samples with actual results.

        """

        progress = kwargs.get("progress_bar", False)
        classifier = GenderClassifier()
        for sample in kwargs['raw_data']:
            
            if sample.task =='question-answering':
                    if "perturbed_context" in sample.__annotations__:
                         genders = [classifier.predict(sample.original_context)
                           for sample in kwargs['raw_data']]
                    else:
                        genders = [classifier.predict(sample.original_question)
                           for sample in kwargs['raw_data']]

            else:
                genders = [classifier.predict(sample.original)
                           for sample in kwargs['raw_data']]

        gender_counts = {
               "male": len([x for x in genders if x == "male"]),
               "female": len([x for x in genders if x == "female"]),
               "unknown": len([x for x in genders if x == "unknown"])
            }

        total_samples = len(kwargs['raw_data'])

        for sample in sample_list:
             if progress:
                progress.update(1)
             
             if sample.test_type == "min_gender_representation_proportion":
                    sample.actual_results = MinScoreOutput(
                        min_score=round(gender_counts[sample.test_case]/total_samples, 2))
                    sample.state = "done"
             
             elif sample.test_type == "min_gender_representation_count":
                    sample.actual_results = MinScoreOutput(
                        min_score=round(gender_counts[sample.test_case], 2))
                    sample.state = "done"
        return sample_list


class EthnicityRepresentation(BaseRepresentation):

    """
    Subclass of BaseRepresentation that implements the ethnicity representation test.

    Attributes:
        alias_name (List[str]): The list of test names that identify the representation measure.

    """

    alias_name = [
        "min_ethnicity_name_representation_count",
        "min_ethnicity_name_representation_proportion"
    ]
    
    supported_tasks = ["ner", "text-classification","question-answering","summarization"]

    def transform(test, data, params):
        """
        Args:
            test (str): name of the test
            data (List[Sample]): The input data to be evaluated for representation test.
            params : parameters specified in config.

        Raises:
            ValueError: If sum of specified proportions in config is greater than 1

        Returns:
            List[Sample]: Ethnicity Representation test results.
        """
        sample_list = []

        if test == "min_ethnicity_name_representation_count":
            if not params:
                expected_representation = {"black": 10, "asian": 10, "white": 10,
                                           "native_american": 10, "hispanic": 10, "inter_racial": 10}

            else:
                if isinstance(params['min_count'], dict):
                    expected_representation = params['min_count']

                elif isinstance(params['min_count'], int):
                    expected_representation = {
                        key: params['min_count'] for key in default_ehtnicity_representation}

            for key, value in expected_representation.items():
                if hasattr(data[0], 'task') and data[0].task == 'question-answering':
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_ethnicity_name_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    sample_list.append(sample)
                     
                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_ethnicity_name_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    sample_list.append(sample)

        if test == "min_ethnicity_name_representation_proportion":
            if not params:
                expected_representation = {"black": 0.13, "asian": 0.13, "white": 0.13,
                                           "native_american": 0.13, "hispanic": 0.13, "inter_racial": 0.13}

            else:
                if isinstance(params['min_proportion'], dict):
                    expected_representation = params['min_proportion']

                    if sum(expected_representation.values()) > 1:
                        print('''Sum of proportions cannot be greater than 1.
                            So min_ethnicity_name_representation_proportion test cannot run''')
                        raise ValueError()

                elif isinstance(params['min_proportion'], float):
                    expected_representation = {
                        key: params['min_proportion'] for key in default_ehtnicity_representation}
                    if sum(expected_representation.values()) > 1:
                        print('''Sum of proportions cannot be greater than 1.
                        So min_ethnicity_name_representation_proportion test cannot run''')
                        raise ValueError()

            for key, value in expected_representation.items(): 
                 if hasattr(data[0], 'task') and data[0].task == 'question-answering':
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_ethnicity_name_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    sample_list.append(sample)
                 
                 else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_ethnicity_name_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    sample_list.append(sample)

        return sample_list

    async def run(sample_list: List[Sample], model: ModelFactory, **kwargs) -> List[Sample]:

        """ 
        Computes the actual results for the enthicity representation test.

        Args:
            sample_list (List[Sample]): The input data to be evaluated for representation test.
            model (ModelFactory): The model to be used for evaluation.

        Returns:
            List[Sample]: The list of samples with actual results.
        """
        progress = kwargs.get("progress_bar", False)

        entity_representation = get_ethnicity_representation_dict(
            kwargs['raw_data'])

        for sample in sample_list:
            if sample.test_type == "min_ethnicity_name_representation_proportion":
                entity_representation_proportion = get_entity_representation_proportions(
                    entity_representation)
                actual_representation = {
                    **default_ehtnicity_representation, **entity_representation_proportion}

                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2))
                sample.state = "done"

            elif sample.test_type == "min_ethnicity_name_representation_count":
                actual_representation = {
                    **default_ehtnicity_representation, **entity_representation}
                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2))
                sample.state = "done"

            if progress:
                progress.update(1)

        return sample_list


class LabelRepresentation(BaseRepresentation):
    """
    Subclass of BaseRepresentation that implements the label representation test.

    Attributes:
        alias_name (List[str]): The list of test names that identify the representation measure.

    """

    alias_name = [
        "min_label_representation_count",
        "min_label_representation_proportion"
    ]
    
    supported_tasks = ["ner", "text-classification"]

    def transform(test, data, params):
        """
        Args:
            test (str): name of the test
            data (List[Sample]): The input data to be evaluated for representation test.
            params : parameters specified in config.

        Raises:
            ValueError: If sum of specified proportions in config is greater than 1

        Returns:
            List[Sample]: Label Representation test results.
        """
        sample_list = []
        labels = [s.expected_results.predictions for s in data]
        if isinstance(data[0].expected_results, NEROutput):
            labels = [x.entity.split("-")[-1]
                      for sentence in labels for x in sentence]
        elif isinstance(data[0].expected_results, SequenceClassificationOutput):
            labels = [x.label for sentence in labels for x in sentence]
        labels = set(labels)

        if test == "min_label_representation_count":

            if not params:
                expected_representation = {k: 10 for k in labels}

            else:
                if isinstance(params['min_count'], dict):
                    expected_representation = params['min_count']

                elif isinstance(params['min_count'], int):
                    expected_representation = {
                        key: params['min_count'] for key in labels}

            for key, value in expected_representation.items():
                sample = MinScoreSample(
                    original="-",
                    category="representation",
                    test_type="min_label_representation_count",
                    test_case=key,
                    expected_results=MinScoreOutput(min_score=value)
                )
                sample_list.append(sample)

        if test == "min_label_representation_proportion":
            if not params:
                expected_representation = {k: (1/len(k))*0.8 for k in labels}

            else:
                if isinstance(params['min_proportion'], dict):
                    expected_representation = params['min_proportion']

                    if sum(expected_representation.values()) > 1:
                        print('''Sum of proportions cannot be greater than 1.
                        So min_label_representation_proportion test cannot run''')
                        raise ValueError()

                elif isinstance(params['min_proportion'], float):
                    expected_representation = {
                        key: params['min_proportion'] for key in labels}
                    if sum(expected_representation.values()) > 1:
                        print('''Sum of proportions cannot be greater than 1.
                        So min_label_representation_proportion test cannot run''')
                        raise ValueError()

            for key, value in expected_representation.items():

                sample = MinScoreSample(
                    original="-",
                    category="representation",
                    test_type=test,
                    test_case=key,
                    expected_results=MinScoreOutput(min_score=value)
                )
                sample_list.append(sample)

        return sample_list

    async def run(sample_list: List[Sample], model: ModelFactory, **kwargs) -> List[Sample]:

        """
        Computes the actual representation of the labels in the dataset.

        Args:
            sample_list (List[Sample]): The input data to be evaluated for representation test.
            model (ModelFactory): The model to be evaluated.

        Returns:
            List[Sample]: Label Representation test results.
        """
        progress = kwargs.get("progress_bar", False)

        entity_representation = get_label_representation_dict(
            kwargs['raw_data'])

        for sample in sample_list:
            if progress:
                progress.update(1)

            if sample.test_type == "min_label_representation_proportion":
                entity_representation_proportion = get_entity_representation_proportions(
                    entity_representation)
                actual_representation = {**entity_representation_proportion}
                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2))
                sample.state = "done"
            elif sample.test_type == "min_label_representation_count":
                actual_representation = {**entity_representation}
                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2))
                sample.state = "done"

        return sample_list


class ReligionRepresentation(BaseRepresentation):
    """
    Subclass of BaseRepresentation that implements the religion representation test.

    Attributes:
        alias_name (List[str]): The list of test names that identify the representation measure.

    """

    alias_name = [
        "min_religion_name_representation_count",
        "min_religion_name_representation_proportion"
    ]
    supported_tasks = ["ner", "text-classification","question-answering","summarization"]

    def transform(test, data, params):
        """
        Args:
            test (str): name of the test
            data (List[Sample]): The input data to be evaluated for representation test.
            params : parameters specified in config.

        Raises:
            ValueError: If sum of specified proportions in config is greater than 1

        Returns:
            List[Sample]: Religion Representation test results.
        """
        sample_list = []

        if test == "min_religion_name_representation_count":
            if not params:
                expected_representation = {
                    'muslim': 5, 'hindu': 5, 'sikh': 5, 'christian': 5, 'jain': 5, 'buddhist': 5, 'parsi': 5}

            else:
                if isinstance(params['min_count'], dict):
                    expected_representation = params['min_count']

                elif isinstance(params['min_count'], int):
                    expected_representation = {
                        key: params['min_count'] for key in default_religion_representation}
                    

            for key, value in expected_representation.items():
                if hasattr(data[0], 'task') and data[0].task == 'question-answering':
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_religion_name_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    sample_list.append(sample)
                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_religion_name_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    sample_list.append(sample)

        if test == "min_religion_name_representation_proportion":
            if not params:
                expected_representation = {'muslim': 0.11, 'hindu': 0.11, 'sikh': 0.11,
                                           'christian': 0.11, 'jain': 0.11, 'buddhist': 0.11, 'parsi': 0.11}

            else:
                if isinstance(params['min_proportion'], dict):
                    expected_representation = params['min_proportion']

                    if sum(expected_representation.values()) > 1:
                        print('''Sum of proportions cannot be greater than 1.
                        So min_religion_name_representation_proportion test cannot run''')
                        raise ValueError()

                elif isinstance(params['min_proportion'], float):
                    expected_representation = {
                        key: params['min_proportion'] for key in default_religion_representation}
                    if sum(expected_representation.values()) > 1:
                        print('''Sum of proportions cannot be greater than 1.
                        So min_religion_name_representation_proportion test cannot run''')
                        raise ValueError()

            entity_representation = get_religion_name_representation_dict(data)
            entity_representation_proportion = get_entity_representation_proportions(
                entity_representation)
            actual_representation = {
                **default_religion_representation, **entity_representation_proportion}
            for key, value in expected_representation.items():
                
                if hasattr(data[0], 'task') and data[0].task == 'question-answering':
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_religion_name_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    sample_list.append(sample)
                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_religion_name_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                        actual_results=MinScoreOutput(
                            min_score=actual_representation[key]),
                        state="done"
                    )
                    sample_list.append(sample)

        return sample_list

    async def run(sample_list: List[Sample], model: ModelFactory, **kwargs) -> List[Sample]:

        """
        Computes the actual representation of religion names in the data.

        Args:
            sample_list (List[Sample]): The input data to be evaluated for representation test.
            model (ModelFactory): The model to be evaluated.

        Returns:
            List[Sample]: Religion Representation test results.

        """

        progress = kwargs.get("progress_bar", False)

        entity_representation = get_religion_name_representation_dict(
            kwargs['raw_data'])

        for sample in sample_list:
            if sample.test_type == "min_religion_name_representation_proportion":
                entity_representation_proportion = get_entity_representation_proportions(
                    entity_representation)
                actual_representation = {
                    **default_religion_representation, **entity_representation_proportion}

                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2))
                sample.state = "done"

            elif sample.test_type == "min_religion_name_representation_count":
                actual_representation = {
                    **default_religion_representation, **entity_representation}

                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2))
                sample.state = "done"

            if progress:
                progress.update(1)

        return sample_list


class CountryEconomicRepresentation(BaseRepresentation):
    """
    Subclass of BaseRepresentation that implements the country economic representation test.

    Attributes:
        alias_name (List[str]): The list of test names that identify the representation measure.

    """

    alias_name = [
        "min_country_economic_representation_count",
        "min_country_economic_representation_proportion"
    ]
    
    supported_tasks = ["ner", "text-classification","question-answering","summarization"]

    def transform(test, data, params):
        """
        Args:
            test (str): name of the test
            data (List[Sample]): The input data to be evaluated for representation test.
            params : parameters specified in config.

        Raises:
            ValueError: If sum of specified proportions in config is greater than 1

        Returns:
            List[Sample]: Country Economic Representation test results.
        """
        sample_list = []

        if test == "min_country_economic_representation_count":
            if not params:
                expected_representation = {
                    'high_income': 10, 'low_income': 10, 'lower_middle_income': 10, 'upper_middle_income': 10}

            else:
                if isinstance(params['min_count'], dict):
                    expected_representation = params['min_count']

                elif isinstance(params['min_count'], int):
                    expected_representation = {
                        key: params['min_count'] for key in default_economic_country_representation}

            for key, value in expected_representation.items():
                if hasattr(data[0], 'task') and data[0].task == 'question-answering':
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_country_economic_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    sample_list.append(sample)
                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_country_economic_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    sample_list.append(sample)

        if test == "min_country_economic_representation_proportion":
            if not params:
                expected_representation = {'high_income': 0.20, 'low_income': 0.20,
                                           'lower_middle_income': 0.20, 'upper_middle_income': 0.20}

            else:
                if isinstance(params['min_proportion'], dict):
                    expected_representation = params['min_proportion']

                    if sum(expected_representation.values()) > 1:
                        print('''Sum of proportions cannot be greater than 1.
                        So min_country_economic_representation_proportion test cannot run''')
                        raise ValueError()

                elif isinstance(params['min_proportion'], float):
                    expected_representation = {
                        key: params['min_proportion'] for key in default_economic_country_representation}
                    if sum(expected_representation.values()) > 1:
                        print('''Sum of proportions cannot be greater than 1. 
                        So min_country_economic_representation_proportion test cannot run''')
                        raise ValueError()

            for key, value in expected_representation.items():
                
                if hasattr(data[0], 'task') and data[0].task == 'question-answering':
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_country_economic_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    sample_list.append(sample)
                
                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_country_economic_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value)
                    )
                    sample_list.append(sample)

        return sample_list

    async def run(sample_list: List[Sample], model: ModelFactory, **kwargs) -> List[Sample]:
        """
        Computes the actual results for the country economic representation test.

        Args:
            sample_list (List[Sample]): The input data to be evaluated for representation test.
            model (ModelFactory): The model to be used for evaluation.

        Returns:
            List[Sample]: Country Economic Representation test results.

        """
        progress = kwargs.get("progress_bar", False)

        entity_representation = get_country_economic_representation_dict(
            kwargs['raw_data'])

        for sample in sample_list:
            if sample.test_type == "min_country_economic_representation_proportion":
                entity_representation_proportion = get_entity_representation_proportions(
                    entity_representation)
                actual_representation = {
                    **default_economic_country_representation, **entity_representation_proportion}

                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2))
                sample.state = "done"

            elif sample.test_type == "min_country_economic_representation_count":
                actual_representation = {
                    **default_economic_country_representation, **entity_representation}

                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2))
                sample.state = "done"

            if progress:
                progress.update(1)

        return sample_list
