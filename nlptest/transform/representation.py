

from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from nlptest.utils.custom_types import Sample
from .utils import default_representation ,default_ehtnicity_representation, white_names, black_names, hispanic_names, asian_names, native_american_names, inter_racial_names, religion_wise_names, default_religion_representation

class BaseRepresentation(ABC):

    """
    Abstract base class for implementing representation measures.

    Attributes:
        alias_name (str): A name or list of names that identify the representation measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented representation measure.
    """

    @staticmethod
    @abstractmethod
    def transform(self):

        """
        Abstract method that implements the representation measure.

        Args:
            data (List[Sample]): The input data to be transformed.

        Returns:
            Any: The transformed data based on the implemented representation measure.
        """

        return NotImplementedError
    
    alias_name = None


class GenderRepresentation(BaseRepresentation):

    alias_name = [
        "min_gender_representation_count",
        "min_gender_representation_proportion"
    ]
    
    def transform(data: List[Sample]):
        return super().transform()


class EthnicityRepresentation(BaseRepresentation):
    
    alias_name = [
        "min_ethnicity_name_representation_count",
        "min_ethnicity_name_representation_proportion"
    ]

    def transform(
            dataset: List[Sample],
            min_count: dict = default_ehtnicity_representation
    ) -> pd.DataFrame():
     
        # define a function to check if a word belongs to any of the given name lists
        def check_name(word, name_lists):
            return any(word.lower() in [name.lower() for name in name_list] for name_list in name_lists)

        # initialize a dictionary to store the ethnicity representation
        ethnicity_representation = {"black": 0, "asian": 0, "white": 0, "native_american": 0, "hispanic": 0, "inter_racial": 0}
        representation_dict = min_count
        
        # iterate over the samples in the dataset
        for sample in dataset:
            # iterate over the expected results in the sample
            for i in sample.expected_results.predictions:
                # check if the word belongs to any of the name lists and update the ethnicity representation
                if check_name(i.span.word, [white_names['first_names'], white_names['last_names']]):
                    ethnicity_representation["white"] += 1
                if check_name(i.span.word, [black_names['first_names'], black_names['last_names']]):
                    ethnicity_representation["black"] += 1
                if check_name(i.span.word, [hispanic_names['first_names'], hispanic_names['last_names']]):
                    ethnicity_representation["hispanic"] += 1
                if check_name(i.span.word, [asian_names['first_names'], asian_names['last_names']]):
                    ethnicity_representation["asian"] += 1
                if check_name(i.span.word, [inter_racial_names['last_names']]):
                    ethnicity_representation["inter_racial"] += 1
                if check_name(i.span.word, [native_american_names['last_names']]):
                    ethnicity_representation["native_american"] += 1


        expected_representation = {**default_ehtnicity_representation, **representation_dict}
        actual_representation = {**default_ehtnicity_representation, **ethnicity_representation}
    
        try:
            ethnicity_representation_df = pd.DataFrame({"Category":"Representation","Test_type":"ethnicity_representation","Original":"-","Test_Case":[label for label in ethnicity_representation.keys()],"expected_result":[value for value in expected_representation.values()],
                                          "actual_result":[value for value in actual_representation.values()]})
         
        except:
              raise ValueError(f"Check your labels. By default, we use these labels only : 'black', 'white', 'hispanic', 'inter_racial', 'asian_pacific_islander' and 'american_indian_alaskan' \n You provided : {representation_dict.keys()}")
        ethnicity_representation_df = ethnicity_representation_df.assign(is_pass=ethnicity_representation_df.apply(lambda row: row['actual_result'] >= row['expected_result'], axis=1))
        return ethnicity_representation_df
    
    

class LabelRepresentation(BaseRepresentation):
    
    alias_name = [
        "min_label_representation_count",
        "min_label_representation_proportion"
    ]

    def transform(data: List[Sample], min_count: dict = default_representation):
        representation_dict = min_count
        entity_representation={}
     
        for sample in data:
            for i in sample.expected_results.predictions:
              if i.entity=='O':
                if  i.entity not in entity_representation:
                  entity_representation[i.entity]=1
                else:
                  entity_representation[i.entity]+=1
             
              elif i.entity in ['B-LOC','I-LOC','B-PER','I-PER','B-MISC','I-MISC','B-ORG','I-ORG']:
                if  i.entity.split("-")[1] not in entity_representation :
                  entity_representation[i.entity.split("-")[1]]=1
                else:
                  entity_representation[i.entity.split("-")[1]]+=1
        expected_representation = {**default_representation, **representation_dict}
        actual_representation = {**default_representation, **entity_representation}
        try:
            label_representation_df = pd.DataFrame({"Category":"Representation","Test_type":"label_representation","Original":"-","Test_Case":[label for label in entity_representation.keys()],"expected_result":[value for value in expected_representation.values()],
                                          "actual_result":[value for value in actual_representation.values()]})
         
        except:
              raise ValueError(f"Check your labels. By default, we use these labels only : 'O', 'LOC', 'PER', 'MISC', 'ORG' \n You provided : {representation_dict.keys()}")
        label_representation_df = label_representation_df.assign(is_pass=label_representation_df.apply(lambda row: row['actual_result'] >= row['expected_result'], axis=1))
        return label_representation_df

class ReligionRepresentation(BaseRepresentation):
    
    alias_name = [
        "min_religion_name_representation_count",
        "min_religion_name_representation_proportion"
    ]

    def transform(
            dataset: List[Sample],
            min_count: dict = default_religion_representation
    ) -> pd.DataFrame():
   
        def check_name(word, name_lists):
            return any(word.lower() in [name.lower() for name in name_list] for name_list in name_lists)

        religion_representation = {'muslim': 0, 'hindu':0, 'sikh':0, 'christian':0, 'jain':0, 'buddhist':0, 'parsi':0}
        representation_dict = min_count
        
        # iterate over the samples in the dataset
        for sample in dataset:
            for i in sample.expected_results.predictions:
                if check_name(i.span.word, [religion_wise_names['Muslim']]):
                    religion_representation["muslim"] += 1
                if check_name(i.span.word, [religion_wise_names['Hindu']]):
                    religion_representation["hindu"] += 1
                if check_name(i.span.word, [religion_wise_names['Sikh']]):
                    religion_representation["sikh"] += 1
                if check_name(i.span.word, [religion_wise_names['Parsi']]):
                    religion_representation["parsi"] += 1
                if check_name(i.span.word, [religion_wise_names['Christian']]):
                    religion_representation["christian"] += 1
                if check_name(i.span.word, [religion_wise_names['Buddhist']]):
                    religion_representation["buddhist"] += 1
                if check_name(i.span.word, [religion_wise_names['Jain']]):
                    religion_representation["jain"] += 1

        expected_representation = {**default_religion_representation, **representation_dict}
        actual_representation = {**default_religion_representation, **religion_representation}
    
        try:
            religion_representation_df = pd.DataFrame({"Category":"Representation","Test_type":"religion_representation","Original":"-","Test_Case":[label for label in religion_representation.keys()],"expected_result":[value for value in expected_representation.values()],
                                          "actual_result":[value for value in actual_representation.values()]})
         
        except:
              raise ValueError(f"Check your labels. By default, we use these labels only : 'muslim', 'hindu', 'sikh', 'christian', 'jain', 'parsi' and 'buddhist' \n You provided : {representation_dict.keys()}")
        religion_representation_df = religion_representation_df.assign(is_pass=religion_representation_df.apply(lambda row: row['actual_result'] >= row['expected_result'], axis=1))
        return religion_representation_df

class CountryEconomicRepresentation(BaseRepresentation):
    
    alias_name = [
        "min_country_economic_representation_count",
        "min_country_economic_representation_proportion"
    ]

    def transform(data: List[Sample]):
        return super().transform()