from collections import defaultdict
from typing import Dict, List
import pandas as pd
from langtest.utils.custom_types import NERPrediction, Sample, SequenceLabel, NEROutput, SequenceClassificationOutput         
from .constants import (
    country_economic_dict, 
    religion_wise_names ,
    white_names ,
    black_names ,
    hispanic_names ,
    asian_names ,
    inter_racial_names ,
    native_american_names 
)
from .custom_bias import add_custom_data



class RepresentationOperation:

    @staticmethod
    def add_custom_representation(data, name, append,task):
        """
        Add custom representation to the given data.

        Args:
            data (list): The data to which the custom representation will be added.
            name (str): The name of the custom representation.
            append (bool): Indicates whether to append the custom representation or replace the existing representation.
            task :  "representation"
        Returns:
            None
        """
        add_custom_data(data, name, append,task)
    
 
            

    def get_label_representation_dict(data: List[Sample]) -> Dict[str, int]:
        """
        Args:
            data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
            dict: a dictionary containing label representation information.
        """

        label_representation = defaultdict(int)

        for sample in data:
            for prediction in sample.expected_results.predictions:
                if isinstance(prediction, SequenceLabel):
                    label_representation[prediction.label] += 1
                elif isinstance(prediction, NERPrediction):
                    if prediction.entity == 'O':
                        label_representation[prediction.entity] += 1
                    elif prediction.entity in ['B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG']:
                        label_representation[prediction.entity.split("-")[1]] += 1

        return label_representation


    def get_country_economic_representation_dict(data: List[Sample]) -> Dict[str, int]:
        """
        Args:
        data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
        Dict[str, int]: a dictionary containing country economic representation information.
        """

        country_economic_representation = {"high_income": 0, "low_income": 0, "lower_middle_income": 0,
                                        "upper_middle_income": 0}
        
        income_mapping = {
        "High-income": "high_income",
        "Lower-middle-income": "low_income",
        "Low-income": "low_income",
        "Upper-middle-income": "upper_middle_income"
        }

        for sample in data:
            if isinstance(sample.expected_results, NEROutput):
                words = [x.span.word.lower() for x in sample.expected_results.predictions]  
            elif isinstance(sample.expected_results, SequenceClassificationOutput):
                words = set(sample.original.replace('.', '').lower().split())
            elif sample.task =='question-answering':
                if "perturbed_context" in sample.__annotations__:  
                    words = set(sample.original_context.replace('.', '').lower().split())
                else:
                    words = set(sample.original_question.replace('.', '').lower().split())
            elif sample.task =='summarization':
                words = set(sample.original.replace('.', '').lower().split())

            for income, countries in country_economic_dict.items():
                for country in countries:
                    country_words = set(country.lower().split()) 
                    if country_words.issubset(words):    
                        country_economic_representation[income_mapping[income]] += 1

        return country_economic_representation


    def get_religion_name_representation_dict(data: List[Sample]) -> Dict[str, int]:
        """
        Args:
            data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
            Dict[str, int]: a dictionary containing religion representation information.
        """

        religion_representation = {'muslim': 0, 'hindu': 0, 'sikh': 0, 'christian': 0, 'jain': 0, 'buddhist': 0, 'parsi': 0}
        religions = ['Muslim', 'Hindu', 'Sikh', 'Parsi', 'Christian', 'Buddhist', 'Jain']

        for sample in data:
            if isinstance(sample.expected_results, NEROutput):
                words = [x.span.word for x in sample.expected_results.predictions]
            elif isinstance(sample.expected_results, SequenceClassificationOutput):
                words = sample.original.split()
            elif sample.task =='question-answering':
                if "perturbed_context" in sample.__annotations__:  
                    words = sample.original_context.split()
                else:
                    words = sample.original_question.split()   
            elif sample.task =='summarization':
                words = sample.original.split()
                
            for i in words:
                for religion in religions:
                    if check_name(i, [religion_wise_names[religion]]):
                        religion_representation[religion.lower()] += 1

        return religion_representation


    def get_ethnicity_representation_dict(data: List[Sample]) -> Dict[str, int]:
        """
        Args:
            data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
            Dict[str, int]: a dictionary containing ethnicity representation information.
        """
        ethnicity_representation = {"black": 0, "asian": 0, "white": 0, "native_american": 0, "hispanic": 0,
                                    "inter_racial": 0}

        for sample in data:
            if isinstance(sample.expected_results, NEROutput):
                words = [x.span.word for x in sample.expected_results.predictions]
            elif isinstance(sample.expected_results, SequenceClassificationOutput):
                words = sample.original.split()
            elif sample.task =='question-answering':
                if "perturbed_context" in sample.__annotations__:  
                    words = sample.original_context.split()
                else:
                    words = sample.original_question.split()   
            elif sample.task =='summarization':
                words = sample.original.split()
                
            for i in words:
                if check_name(i, [white_names['first_names'], white_names['last_names']]):
                    ethnicity_representation["white"] += 1
                if check_name(i, [black_names['first_names'], black_names['last_names']]):
                    ethnicity_representation["black"] += 1
                if check_name(i, [hispanic_names['first_names'], hispanic_names['last_names']]):
                    ethnicity_representation["hispanic"] += 1
                if check_name(i, [asian_names['first_names'], asian_names['last_names']]):
                    ethnicity_representation["asian"] += 1
                if check_name(i, [inter_racial_names['last_names']]):
                    ethnicity_representation["inter_racial"] += 1
                if check_name(i, [native_american_names['last_names']]):
                    ethnicity_representation["native_american"] += 1

        return ethnicity_representation


    def get_entity_representation_proportions(entity_representation):
        """
        Args:
        entity_representation (dict): a dictionary containing representation information.

        Returns:
        dict: a dictionary with proportions of each entity.
        """

        total_entities = sum(entity_representation.values())
        entity_representation_proportion = {}
        for k, v in entity_representation.items():
            if total_entities == 0:
                entity_representation_proportion[k] = 0
            else:
                entity_representation_proportion[k] = v / total_entities

        return entity_representation_proportion   
    





def get_substitution_names(values_list: List[List[str]]) -> List[str]:
    """ Helper function to get list of substitution names

    Args:
         values_list (List[List[str]]):
            list of substitution lists.

    Returns:
         List[str]:
            List of substitution names
    """
    substitution_names = []
    for lst in values_list:
        substitution_names.extend(lst)

    return substitution_names


def create_terminology(ner_data: pd.DataFrame) -> Dict[str, List[str]]:
    """Iterate over the DataFrame to create terminology from the predictions. IOB format converted to the IO.

    Args:
        ner_data: Pandas DataFrame that has 2 column, 'text' as string and 'label' as list of labels

    Returns:
        Dictionary of entities and corresponding list of words.
    """
    terminology = {}

    chunk = list()
    ent_type = None
    for i, row in ner_data.iterrows():

        sent_labels = row.label
        for token_indx, label in enumerate(sent_labels):

            if label.startswith('B'):

                if chunk:
                    if terminology.get(ent_type, None):
                        terminology[ent_type].append(" ".join(chunk))
                    else:
                        terminology[ent_type] = [" ".join(chunk)]

                sent_tokens = row.text.split(' ')
                chunk = [sent_tokens[token_indx]]
                ent_type = label[2:]

            elif label.startswith('I'):

                sent_tokens = row.text.split(' ')
                chunk.append(sent_tokens[token_indx])

            else:

                if chunk:
                    if terminology.get(ent_type, None):
                        terminology[ent_type].append(" ".join(chunk))
                    else:
                        terminology[ent_type] = [" ".join(chunk)]

                chunk = None
                ent_type = None

    return terminology




def check_name(word: str, name_lists: List[List[str]]) -> bool:
    """
    Checks if a word is in a list of list of strings

    Args:
        word (str):
            string to look for
        name_lists (List[List[str]]):
            list of lists of potential candidates
    """
    return any(word.lower() in [name.lower() for name in name_list] for name_list in name_lists)


