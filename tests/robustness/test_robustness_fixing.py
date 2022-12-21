import unittest
import os
from nlptest.robustness.robustness_fixing import augment_robustness


class TestRobustnessFixing(unittest.TestCase):

    def setUp(self):
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))

        # Set the test file path
        self.test_file_path = os.path.join(THIS_DIR, os.pardir, 'resources/test.conll')

        # Set the augmented file path
        self.augmented_file_path = os.path.join(THIS_DIR, os.pardir, 'resources/augmented_test.conll')

        # Delete the augmented file if it has been created already
        if os.path.exists(self.augmented_file_path):
            os.remove(self.augmented_file_path)

        # Set perturbation map
        self.perturbation_map = {
            "capitalization_upper": 0.05,
            "capitalization_lower": 0.05,
            "capitalization_title": 0.05,
            "add_punctuation": 0.05,
            "strip_punctuation": 0.05,
            "introduce_typos": 0.05,
            "add_contractions": 0.05,
            "american_to_british": 0.05,
            "add_context": 0.05,
            "swap_entities": 0.05,
            "swap_cohyponyms": 0.05
        }

        # Set entity_perturbation_map
        self.entity_perturbation_map = {
            "capitalization_upper": {'PER': 0.05, 'ORG': 0.02, 'LOC': 0.06, 'MISC': 0.01},
            "capitalization_lower": {'PER': 0.05, 'ORG': 0.02, 'LOC': 0.06, 'MISC': 0.01},
            'capitalization_title': {'PER': 0.05, 'ORG': 0.02, 'LOC': 0.06, 'MISC': 0.01},
            'add_punctuation': {'PER': 0.05, 'ORG': 0.02, 'LOC': 0.06, 'MISC': 0.01},
            'strip_punctuation': {'PER': 0.05, 'ORG': 0.02, 'LOC': 0.06, 'MISC': 0.01},
            'introduce_typos': {'PER': 0.05, 'ORG': 0.02, 'LOC': 0.06, 'MISC': 0.01},
            'add_contractions': {'PER': 0.05, 'ORG': 0.02, 'LOC': 0.06, 'MISC': 0.01},
            'american_to_british': {'PER': 0.05, 'ORG': 0.02, 'LOC': 0.06, 'MISC': 0.01},
            'add_context': {'PER': 0.05, 'ORG': 0.02, 'LOC': 0.06, 'MISC': 0.01},
            'swap_entities': {'PER': 0.05, 'ORG': 0.02, 'LOC': 0.06, 'MISC': 0.01},
            'swap_cohyponyms': {'PER': 0.05, 'ORG': 0.02, 'LOC': 0.06, 'MISC': 0.01}
        }

    def test_no_perturb_map(self):
        # Test error when no perturbation map passed
        with self.assertRaises(ValueError):
            augment_robustness(conll_path=self.test_file_path,
                               conll_save_path=self.augmented_file_path,
                               print_info=False,
                               ignore_warnings=True,
                               random_state=0)

    def test_both_perturb_maps_passed(self):
        # Test error when both perturbation maps passed
        with self.assertRaises(ValueError):
            augment_robustness(conll_path=self.test_file_path,
                               conll_save_path=self.augmented_file_path,
                               perturbation_map=self.perturbation_map,
                               entity_perturbation_map=self.entity_perturbation_map,
                               print_info=False,
                               ignore_warnings=True,
                               random_state=0)

    def test_perturbation_map_works(self):
        augment_robustness(conll_path=self.test_file_path,
                           conll_save_path=self.augmented_file_path,
                           perturbation_map=self.perturbation_map,
                           print_info=False,
                           ignore_warnings=True,
                           random_state=0)

        self.assertTrue(os.path.exists(self.augmented_file_path))

    def test_entity_perturbation_map_works(self):
        augment_robustness(conll_path=self.test_file_path,
                           conll_save_path=self.augmented_file_path,
                           entity_perturbation_map=self.entity_perturbation_map,
                           print_info=False,
                           ignore_warnings=True,
                           random_state=0)

        self.assertTrue(os.path.exists(self.augmented_file_path))


if __name__ == '__main__':
    unittest.main()
