from abc import ABC, abstractmethod
import random
from typing import List

from ..utils.custom_types import Sample
from .utils import male_pronouns, female_pronouns, neutral_pronouns


class BaseBias(ABC):

    @abstractmethod
    def transform(self):
        return NotImplementedError

    alias_name = None

    
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
                replace_token = random.choice(tokens_to_substitute)
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
                      if replace_token in value:
                        type_of_pronoun = str(key)
                        break

                chosen_token= random.choice(chosen_dict[type_of_pronoun])
                replaced_string = sample.original.replace(replace_token, chosen_token)
                sample.test_case = replaced_string
            else:
              sample.test_case = sample.original
            
            sample.category="Bias"
      
        return sample_list

