

from abc import ABC, abstractmethod, abstractstaticmethod
import random
import re
import pandas as pd
from typing import List, Optional

from nlptest.datahandler.datasource import DataFactory
from nlptest.transform.perturbation import PerturbationFactory
from nlptest.transform.utils import DEFAULT_PERTURBATIONS


class BaseAugmentaion(ABC):

    @abstractmethod
    def fix(self):
        return NotImplementedError


class AugmentRobustness(BaseAugmentaion):

    def fix(
        self,
        input_path:str,
        output_path,
        h_report,
        inplace: bool = False,
        max_prop:float = 0.5,
        config= None,
        regex_pattern: str = "\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])"
    ):
        data = DataFactory(input_path).load()
        config = {list(i.keys())[0] if type(i) == dict else i: i  for i in config['tests_types']}
        # data['pos_tag'] = data['label'].apply(lambda x: ["NN NN"] * len(x))
        # entites = set(j.split("-")[-1] for i in data['label'] for j in i)
        suggest = self.suggestions(h_report)
        sum_propotion = suggest['proportion_increase'].sum()
        if suggest.shape[0] <= 0:
            print("Test metrics all have over 0.9 f1-score for all perturbations. Perturbations will not be applied.")

        fianl_aug_data = []
        # DEFAULT_PERTURBATIONS.remove('swap_entities')
        for proportion in suggest.iterrows():
            test_type = [config.get(proportion[-1]['test_type'])]

            if proportion[-1]['test_type'] in DEFAULT_PERTURBATIONS:
                sample_length = len(data) * max_prop * (proportion[-1]['proportion_increase']/sum_propotion)
                if inplace:
                    hash_map = {k: v for k, v in enumerate(data)}
                    sample_indices = random.sample(range(0, len(data)), int(sample_length))
                    for each in sample_indices:
                        hash_map[each] = PerturbationFactory([hash_map[each]], test_type).transform()[0]
                    fianl_aug_data.extend(list(hash_map.values()))
                else:
                    sample_data = random.choices(data, k=int(sample_length))
                    aug_data = PerturbationFactory(sample_data, test_type).transform()
                    fianl_aug_data.extend(aug_data)
   
        # data.extend(fianl_aug_data)
        self.save(fianl_aug_data, output_path)
        return fianl_aug_data

    def suggestions(self, report):
        
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

    def save(self, data, save_path):
        with open(save_path+"augmenated_train.conll", "w") as fw:
            words = [i.test_case.split() if i.test_case else "" for i in data ]
            fw.write("\n\n".join('-X- -X- \n'.join(ew) for ew in words))
