

from abc import ABC, abstractstaticmethod
import random
import re
import pandas as pd
from typing import List, Optional

from nlptest.datahandler.datasource import DataFactory
from nlptest.transform.perturbation import PerturbationFactory


class BaseAugmentaion(ABC):

    @abstractstaticmethod
    def fix():
        return NotImplementedError


class AugmentRobustness(BaseAugmentaion):

    def fix(
        data_path:str,
        h_report,
        save_path,
        config= None,
        nosie_prob:float = 0.5,
        test: Optional[List[str]] = None,
        optimized_inplace: bool = False,
        regex_pattern: str = "\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])"
    ):
        data = DataFactory(data_path).load()
        config = {list(i.keys())[0] if type(i) == dict else i: i  for i in config['tests_types']}
        # data['pos_tag'] = data['label'].apply(lambda x: ["NN NN"] * len(x))
        # entites = set(j.split("-")[-1] for i in data['label'] for j in i)
        suggest = AugmentRobustness.suggestions(h_report)
        
        if suggest.shape[0] <= 0:
            print("Test metrics all have over 0.9 f1-score for all perturbations. Perturbations will not be applied.")

        fianl_aug_data = []
        for proportion in suggest.iterrows():
            test_type = [config.get(proportion[-1]['test_type'])]
            if optimized_inplace:
                continue
            else:
                sample_length = len(data) * proportion[-1]['proportion_increase']
                sample_data = random.choices(data, k=int(sample_length))
                aug_data = PerturbationFactory(sample_data, test_type).transform()
                fianl_aug_data.extend(aug_data)
   
        # data.extend(fianl_aug_data)
        AugmentRobustness.save(fianl_aug_data, save_path)
        return fianl_aug_data

    def suggestions(report):
        
        def proportion_values(x):
            if x >= 1:
                return None
            elif x > 0.9:
                return 0.05
            elif x > 0.8:
                return 0.1
            elif x > 0.7:
                return 0.2
            else:
                return 0.3
        
        report['ratio'] = report['pass_rate']/ report['minimum_pass_rate']
        report['proportion_increase'] = report['ratio'].apply(
                                            lambda x: proportion_values(x)
                                        )
        return report[~report['proportion_increase'].isna()][['test_type', 'ratio', 'proportion_increase']]

    def save(data, save_path):
        with open(save_path+"augmenated_train.conll", "w") as fw:
            words = [i.test_case.split() if i.test_case else "" for i in data ]
            
            fw.write("\n\n".join('\n'.join(ew) for ew in words))
