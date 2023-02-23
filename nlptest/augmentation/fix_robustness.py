

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

   
        
