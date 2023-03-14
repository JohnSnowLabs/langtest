from abc import ABC, abstractmethod
import random
import re
from typing import List

from ..utils.custom_types import Sample
from .utils import male_pronouns, female_pronouns, neutral_pronouns


class BaseBias(ABC):

    """
    Abstract base class for implementing bias measures.

    Attributes:
        alias_name (str): A name or list of names that identify the bias measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented bias measure.
    """
    alias_name = None


    @abstractmethod
    def transform(self):
        """
        Abstract method that implements the bias measure.

        Args:
            data (List[Sample]): The input data to be transformed.

        Returns:
            Any: The transformed data based on the implemented bias measure.
        """
        return NotImplementedError


    
class GenderPronounBias(BaseBias):
    alias_name = [
        "replace_to_male_pronouns",
        "replace_to_female_pronouns",
        "replace_to_neutral_pronouns"
    ]

    @staticmethod
    def transform(sample_list: List[Sample], pronouns_to_substitute: List[str], pronoun_type:str) -> List[Sample]:
        """Replace pronouns to check the gender bias

        Args:
            sample_list: List of sentences to apply perturbation.
            pronouns_to_substitute: list of pronouns that need to be substituted.
            pronoun_type: replacing pronoun type string ('male', 'female' or 'neutral')

        Returns:
            List of sentences with replaced pronouns
        """


        for sample in sample_list:
          
            tokens_to_substitute = [token for token in sample.original.split(' ') if token.lower() in pronouns_to_substitute]
          
            if len(tokens_to_substitute)!=0:
                replaced_string = None
                for replace_token in tokens_to_substitute:
                  if pronoun_type =="female":
                    combined_dict = {k: male_pronouns[k] + neutral_pronouns[k] for k in male_pronouns.keys()}
                    chosen_dict = female_pronouns
                  elif pronoun_type =="male":
                    combined_dict = {k: female_pronouns[k] + neutral_pronouns[k] for k in female_pronouns.keys()}  
                    chosen_dict = male_pronouns      
                  elif pronoun_type =="neutral":
                    combined_dict = {k: female_pronouns[k] + male_pronouns[k] for k in female_pronouns.keys()}
                    chosen_dict = neutral_pronouns  

                  for key, value in combined_dict.items() :
                        if replace_token.lower() in value:
                          type_of_pronoun = str(key)
                          break
                  
                  chosen_token= random.choice(chosen_dict[type_of_pronoun])

                  if not replaced_string:
                    regex = r'\b{}\b'.format(replace_token)
                    replaced_string = re.sub(regex, chosen_token, sample.original)
              
                  else:
                    regex = r'\b{}\b'.format(replace_token)
                    replaced_string = re.sub(regex, chosen_token, replaced_string)

                  sample.test_case = replaced_string

            else:
              sample.test_case = sample.original
            
            sample.category="Bias"
      
        return sample_list

