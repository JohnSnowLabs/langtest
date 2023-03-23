from typing import Dict, List
import pandas as pd
from ..resources import Resource

resources = Resource()

DEFAULT_PERTURBATIONS = [
    "uppercase",
    "lowercase",
    "titlecase",
    "add_punctuation",
    "strip_punctuation",
    "add_typo",
    "american_to_british",
    "british_to_american",
    "add_context",
    "add_contractions",
    "swap_entities",
    "swap_cohyponyms",
    "replace_to_male_pronouns",
    "replace_to_female_pronouns",
    "replace_to_neutral_pronouns"
]

PERTURB_CLASS_MAP = {
    "uppercase": 'UpperCase',
    "lowercase": 'LowerCase',
    "titlecase": 'TitleCase',
    "add_punctuation": 'AddPunctuation',
    "strip_punctuation": 'StripPunctuation',
    "add_typo": 'AddTypo',
    "american_to_british": 'ConvertAccent',
    "british_to_american": 'ConvertAccent',
    "add_context": 'AddContext',
    "add_contractions": 'AddContraction',
    "swap_entities": 'SwapEntities',
    "swap_cohyponyms": 'SwapCohyponyms',
    "replace_to_male_pronouns": "GenderPronounBias",
    "replace_to_female_pronouns": "GenderPronounBias",
    "replace_to_neutral_pronouns": "GenderPronounBias"
}

# @formatter:off
A2B_DICT = resources['A2B_DICT']
# @formatter:on

TYPO_FREQUENCY = resources['TYPO_FREQUENCY']
CONTRACTION_MAP = resources['CONTRACTION_MAP']

# Curated from the United States Census Bureau surveys
white_names = resources['WHITE_NAMES']
black_names = resources['BLACK_NAMES']
hispanic_names = resources['HISPANIC_NAMES']
native_american_names = resources['NATIVE_AMERICAN_NAMES']
asian_names = resources['ASIAN_NAMES']
inter_racial_names = resources['INTER_RACIAL_NAMES']
religion_wise_names = resources['RELIGION_WISE_NAMES']

# Dicts of respective gender pronouns
female_pronouns = resources['MALE_PRONOUNS']
male_pronouns = resources['FEMALE_PRONOUNS']
neutral_pronouns = resources['NEUTRAL_PRONOUNS']

# Add country economic dict
country_economic_dict = resources['COUNTRY_ECONOMIC_DICT']


def get_substitution_names(values_list):
    """ Helper function to get list of substitution names

    Args:
         values_list : list of substitution lists.

    Returns:
         List of substitution names
    """
    substitution_names = []
    for lst in values_list:
        substitution_names.extend(lst)

    return substitution_names


def create_terminology(ner_data: pd.DataFrame) -> Dict[str, List[str]]:
    """Iterate over the DataFrame to create terminology from the predictions.
    IOB format converted to the IO.

    Args:
        ner_data: Pandas DataFrame that has 2 column, 'text' as string and'label' as list of labels

    Returns:
        Dictionary of entities and corresponding list of words.
    """
    terminology = {}

    chunk = list()
    ent_type = None
    for idx, row in ner_data.iterrows():
        sent_labels = row.label
        for token_indx, label in enumerate(sent_labels):
            if label.startswith('B'):
                if chunk:
                    if terminology.get(ent_type, None):
                        terminology[ent_type].append(" ".join(chunk))
                    else:
                        terminology[ent_type] = [" ".join(chunk)]

                sent_tokens = row.text.split(' ')
                chunk = [sent_tokens[token_indx]]
                ent_type = label[2:]

            elif label.startswith('I'):

                sent_tokens = row.text.split(' ')
                chunk.append(sent_tokens[token_indx])
            else:
                if chunk:
                    if terminology.get(ent_type, None):
                        terminology[ent_type].append(" ".join(chunk))
                    else:
                        terminology[ent_type] = [" ".join(chunk)]
                chunk = None
                ent_type = None
    return terminology
