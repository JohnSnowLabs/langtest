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
    for _, row in ner_data.iterrows():
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


default_label_representation = {
    'O': 0, 'LOC': 0, 'PER': 0, 'MISC': 0, 'ORG': 0}
default_ehtnicity_representation = {
    'black': 0, 'asian': 0, 'white': 0, 'native_american': 0, 'hispanic': 0, 'inter_racial': 0}
default_religion_representation = {
    'muslim': 0, 'hindu': 0, 'sikh': 0, 'christian': 0, 'jain': 0, 'buddhist': 0, 'parsi': 0}
default_economic_country_representation = {
    'high_income': 0,
    'low_income': 0,
    'lower_middle_income': 0,
    'upper_middle_income': 0}


def get_label_representation_dict(data):
    """
    Args:
        data (List[Sample]): The input data to be evaluated for representation test.

    Returns:
        dict: a dictionary containing label representation information.
    """

    entity_representation = {}
    for sample in data:
        for i in sample.expected_results.predictions:
            if i.entity == 'O':
                if i.entity not in entity_representation:
                    entity_representation[i.entity] = 1
                else:
                    entity_representation[i.entity] += 1

            elif i.entity in ['B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG']:
                if i.entity.split("-")[1] not in entity_representation:
                    entity_representation[i.entity.split("-")[1]] = 1
                else:
                    entity_representation[i.entity.split("-")[1]] += 1

    return entity_representation


def check_name(word, name_lists):
    return any(word.lower() in [name.lower() for name in name_list] for name_list in name_lists)


def get_country_economic_representation_dict(data):
    """
    Args:
       data (List[Sample]): The input data to be evaluated for representation test.

    Returns:
       dict: a dictionary containing country economic representation information.
    """

    country_economic_representation = {
        "high_income": 0, "low_income": 0, "lower_middle_income": 0,  "upper_middle_income": 0}

    for sample in data:
        for i in sample.expected_results.predictions:
            if check_name(i.span.word, [country_economic_dict['High-income']]):
                country_economic_representation["high_income"] += 1
            if check_name(i.span.word, [country_economic_dict['Low-income']]):
                country_economic_representation["low_income"] += 1
            if check_name(i.span.word, [country_economic_dict['Lower-middle-income']]):
                country_economic_representation["lower_middle_income"] += 1
            if check_name(i.span.word, [country_economic_dict['Upper-middle-income']]):
                country_economic_representation["upper_middle_income"] += 1

    return country_economic_representation


def get_religion_name_representation_dict(data):
    """
    Args:
        data (List[Sample]): The input data to be evaluated for representation test.

    Returns:
        dict: a dictionary containing religion representation information.
    """

    religion_representation = {'muslim': 0, 'hindu': 0, 'sikh': 0,
                               'christian': 0, 'jain': 0, 'buddhist': 0, 'parsi': 0}

    for sample in data:
        for i in sample.expected_results.predictions:
            if check_name(i.span.word, [religion_wise_names['Muslim']]):
                religion_representation["muslim"] += 1
            if check_name(i.span.word, [religion_wise_names['Hindu']]):
                religion_representation["hindu"] += 1
            if check_name(i.span.word, [religion_wise_names['Sikh']]):
                religion_representation["sikh"] += 1
            if check_name(i.span.word, [religion_wise_names['Parsi']]):
                religion_representation["parsi"] += 1
            if check_name(i.span.word, [religion_wise_names['Christian']]):
                religion_representation["christian"] += 1
            if check_name(i.span.word, [religion_wise_names['Buddhist']]):
                religion_representation["buddhist"] += 1
            if check_name(i.span.word, [religion_wise_names['Jain']]):
                religion_representation["jain"] += 1

    return religion_representation


def get_ethnicity_representation_dict(data):
    """
    Args:
        data (List[Sample]): The input data to be evaluated for representation test.

    Returns:
        dict: a dictionary containing ethnicity representation information.
    """
    ethnicity_representation = {"black": 0, "asian": 0, "white": 0,
                                "native_american": 0, "hispanic": 0, "inter_racial": 0}

    for sample in data:
        for i in sample.expected_results.predictions:
            if check_name(i.span.word, [white_names['first_names'], white_names['last_names']]):
                ethnicity_representation["white"] += 1
            if check_name(i.span.word, [black_names['first_names'], black_names['last_names']]):
                ethnicity_representation["black"] += 1
            if check_name(i.span.word, [hispanic_names['first_names'], hispanic_names['last_names']]):
                ethnicity_representation["hispanic"] += 1
            if check_name(i.span.word, [asian_names['first_names'], asian_names['last_names']]):
                ethnicity_representation["asian"] += 1
            if check_name(i.span.word, [inter_racial_names['last_names']]):
                ethnicity_representation["inter_racial"] += 1
            if check_name(i.span.word, [native_american_names['last_names']]):
                ethnicity_representation["native_american"] += 1

    return ethnicity_representation


def get_entity_representation_proportions(entity_representation):
    """
    Args:
       entity_representation (dict): a dictionary containing representation information.

    Returns:
       dict: a dictionary with proportions of each entity.
    """

    total_entities = sum(entity_representation.values())
    entity_representation_proportion = {}
    for k, v in entity_representation.items():
        entity_representation_proportion[k] = v/total_entities

    return entity_representation_proportion
