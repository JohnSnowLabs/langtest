

from abc import ABC, abstractmethod
from typing import List, Optional

from nlptest.datahandler.datasource import DataFactory


class BaseAugmentaion(ABC):

    @abstractmethod
    @staticmethod
    def fix():
        return NotImplementedError


class AugmentRobustness(BaseAugmentaion):

    def fix(
        data_path:str,
        model,
        h_report,
        random_state:int,
        nosie_prob:float = 0.5,
        test: Optional[List[str]] = None,
        starting_context: Optional[List[str]] = None,
        ending_context: Optional[List[str]] = None,
        optimized_inplace: bool = False,
        
    ):
        data = DataFactory(data_path).load()
        entites = set(j.split("-")[-1] for i in data['label'] for j in i)
        suggest = AugmentRobustness.suggestions(h_report)
        
   
        
    def suggestions(self, report):

            if report['pass rate']/report['min pass rate'] > 1:
                return None
            elif report['pass rate']/report['min pass rate'] > 0.9:
                return 0.05
            elif report['pass rate']/report['min pass rate'] > 0.8:
                return 0.1
            elif report['pass rate']/report['min pass rate'] > 0.7:
                return 0.2
            else:
                return 0.3