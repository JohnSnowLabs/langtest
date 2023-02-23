

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
        
        default_proportion_dict = {
            0.9: None,
            0.75: 0.05,
            0.6: 0.1,
            0.4: 0.2
        }
        report['ratio'] = report['pass rate']/ report['min pass rate']
        report['proportion_increase'] = report['ratio'].apply(
                                            lambda x: default_proportion_dict.get(x, 0.3)
                                        )
        return report[['test type', 'ratio', 'proportion_increase']]