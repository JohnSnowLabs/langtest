

from abc import ABC, abstractmethod, abstractstaticmethod
import random
import re
import pandas as pd
from typing import List, Optional

from nlptest.datahandler.datasource import DataFactory
from nlptest.transform import TestFactory


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
        config,
        inplace: bool = False,
        max_prop:float = 0.5
    ):
        self.df = DataFactory(input_path)
        data = self.df.load()
        supported_tests = TestFactory.test_scenarios()
        self.config = config
        # data['pos_tag'] = data['label'].apply(lambda x: ["NN NN"] * len(x))
        # entites = set(j.split("-")[-1] for i in data['label'] for j in i)
        suggest = self.suggestions(h_report)
        sum_propotion = suggest['proportion_increase'].sum()
        if suggest.shape[0] <= 0:
            print("Test metrics all have over 0.9 f1-score for all perturbations. Perturbations will not be applied.")

        fianl_aug_data = []
        
        for proportion in suggest.iterrows():
            cat = proportion[-1]['category'].lower()
            test = proportion[-1]['test_type'].lower()
            test_type = {
                cat: {
                test: self.config.get('tests').get(cat).get(test)
                }
            }
            if proportion[-1]['test_type'] in supported_tests[cat]:
                sample_length = len(data) * max_prop * (proportion[-1]['proportion_increase']/sum_propotion)
                if inplace:
                    hash_map = {k: v for k, v in enumerate(data)}
                    sample_indices = random.sample(range(0, len(data)), int(sample_length))
                    for each in sample_indices:
                        hash_map[each] = TestFactory.transform([hash_map[each]], test_type)[0]
                    fianl_aug_data.extend(list(hash_map.values()))
                else:
                    sample_data = random.choices(data, k=int(sample_length))
                    aug_data = TestFactory.transform(sample_data, test_type)
                    fianl_aug_data.extend(aug_data)
   
        data.extend(fianl_aug_data)
        self.to_export(data, output_path)
        return fianl_aug_data

    def suggestions(self, report):
        
        report['ratio'] = report['pass_rate']/ report['minimum_pass_rate']
        report['proportion_increase'] = report['ratio'].apply(
                                            lambda x: self._proportion_values(x)
                                        )
        return report[~report['proportion_increase'].isna()][['category','test_type', 'ratio', 'proportion_increase']]

    def to_export(self, data, output_path):
        self.df.export(data, output_path)
        

    def _proportion_values(self, x):
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