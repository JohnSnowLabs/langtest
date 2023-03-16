

from abc import ABC, abstractmethod
from typing import List

from nlptest.utils.custom_types import Sample
from .utils import default_representation ,default_ehtnicity_representation, white_names, black_names, hispanic_names, asian_names, native_american_names, inter_racial_names

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
        """Converts input sentences using a conversion dictionary
        Args:
            sample_list: List of sentences to process.
            strategy: Config method to adjust where will context tokens added. start, end or combined.
            starting_context: list of terms (context) to input at start of sentences.
            ending_context: list of terms (context) to input at end of sentences.
        Returns:
            List of sentences that context added at to begging, end or both, randomly.
        """
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

    def transform(data: List[Sample]):
        return super().transform()

class ReligionRepresentation(BaseRepresentation):
    
    alias_name = [
        "min_religion_name_representation_count",
        "min_religion_name_representation_proportion"
    ]

    def transform(data: List[Sample]):
        return super().transform()

class CountryEconomicRepresentation(BaseRepresentation):
    
    alias_name = [
        "min_country_economic_representation_count",
        "min_country_economic_representation_proportion"
    ]

    def transform(data: List[Sample]):
        return super().transform()