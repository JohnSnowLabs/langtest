

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseAugmentaion(ABC):

    @abstractmethod
    @staticmethod
    def fix():
        return NotImplementedError


class AugmentRobustness(BaseAugmentaion):

    def fix(
        dataset:str,
        model,
        random_state:int,
        nosie_prob:float = 0.5,
        test: Optional[List[str]] = None,
        starting_context: Optional[List[str]] = None,
        ending_context: Optional[List[str]] = None,
        optimized_inplace: bool = False,
        
    ):

        augmentation_report = dict()
        #   Save entity coverage to the augmentation_report
        augmentation_report['entity_coverage'] = dict()
        augmentation_report['augmentation_coverage'] = dict()

   
        
    def suggestions(self, report):

            if report['pass_rate']/report['min_pass_rate'] > 1:
                return None

            elif report['pass_rate']/report['min_pass_rate'] > 0.9:
                return 0.05
            elif report['pass_rate']/report['min_pass_rate'] > 0.8:
                return 0.1
            elif report['pass_rate']/report['min_pass_rate'] > 0.7:
                return 0.2
            else:
                return 0.3