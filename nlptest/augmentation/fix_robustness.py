

from abc import ABC, abstractmethod
import random
import re
import pandas as pd
from typing import List, Optional

from nlptest.datahandler.datasource import DataFactory
from nlptest.transform.perturbation import PerturbationFactory


class BaseAugmentaion(ABC):

    @abstractmethod
    @staticmethod
    def fix():
        return NotImplementedError


class AugmentRobustness(BaseAugmentaion):

    def fix(
        data_path:str,
        h_report,
        random_state:int,
        nosie_prob:float = 0.5,
        test: Optional[List[str]] = None,
        starting_context: Optional[List[str]] = None,
        ending_context: Optional[List[str]] = None,
        optimized_inplace: bool = False,
        regex_pattern: str = "\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])"
    ):
        data = DataFactory(data_path).load()
        data['pos_tag'] = data['label'].apply(lambda x: ["NN NN"] * len(x))
        # data['pos_'] = data['label'].apply(lambda x: ["NN"] * len(x))
        entites = set(j.split("-")[-1] for i in data['label'] for j in i)
        suggest = AugmentRobustness.suggestions(h_report)
        
        if suggest.shape[0] <= 0:
            print("Test metrics all have over 0.9 f1-score for all perturbations. Perturbations will not be applied.")

        if starting_context is None:
            starting_context = ["Description:", "MEDICAL HISTORY:", "FINDINGS:", "RESULTS: ",
                            "Report: ", "Conclusion is that "]
        starting_context = [re.split(regex_pattern, i) for i in starting_context if i != '']

        if ending_context is None:
            ending_context = ["according to the patient's family", "as stated by the patient",
                            "due to various circumstances", "confirmed by px"]
        ending_context = [re.split(regex_pattern, i) for i in ending_context if i != '']

        fianl_aug_data = []
        for proportion in suggest.iterrows():
            if optimized_inplace:
                continue
            else:
                sample_length = len(data) * proportion['proportion_increase']
                sample_data = random.choices(data, k=int(sample_length))
                aug_data = PerturbationFactory(sample_data, [proportion['test type']]).transform()
                fianl_aug_data.extend(aug_data)
   
        
    def suggestions(self, report):
        
        def proportion_values(x):
            if x > 0.9:
                return None
            elif x > 0.75:
                return 0.05
            elif x > 0.6:
                return 0.1
            elif x > 0.4:
                return 0.2
            else:
                return 0.3
        
        report['ratio'] = report['pass rate']/ report['min pass rate']
        report['proportion_increase'] = report['ratio'].apply(
                                            lambda x: proportion_values(x)
                                        )
        return report[['test type', 'ratio', 'proportion_increase']]

    