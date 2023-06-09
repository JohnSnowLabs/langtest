from .utils import (asian_names, black_names, country_economic_dict, hispanic_names, inter_racial_names, male_pronouns,
                    native_american_names, neutral_pronouns, religion_wise_names, white_names,female_pronouns)





def add_custom_data(file, name):
    custom_data = file
    new_name = name

    if new_name in (
            "replace_to_high_income_country",
            "replace_to_low_income_country",
            "replace_to_upper_middle_income_country",
            "replace_to_lower_middle_income_country"
    ):
        valid_names = country_economic_dict.keys()
        # Validate the schema
        if not set(custom_data.keys()).issubset(valid_names):
            raise ValueError(f"Invalid schema. It should be one of: {', '.join(valid_names)}.")

        # Append values to existing keys
        for key, values in custom_data.items():
            country_economic_dict[key] += values

    elif new_name in (
            "replace_to_muslim_names",
            "replace_to_hindu_names",
            "replace_to_christian_names",
            "replace_to_sikh_names",
            "replace_to_jain_names",
            "replace_to_parsi_names",
            "replace_to_buddhist_names"
    ):
        valid_names = religion_wise_names.keys()

        # Validate the schema
        if not set(custom_data.keys()).issubset(valid_names):
            raise ValueError(f"Invalid schema. It should be one of: {', '.join(valid_names)}.")

        # Append values to existing keys
        for key, values in custom_data.items():
            religion_wise_names[key] += values

    elif new_name in (
            "replace_to_white_firstnames",
            "replace_to_black_firstnames",
            "replace_to_hispanic_firstnames",
            "replace_to_asian_firstnames",
            "replace_to_white_lastnames",
            "replace_to_black_lastnames",
            "replace_to_hispanic_lastnames",
            "replace_to_asian_lastnames",
            "replace_to_native_american_lastnames",
            "replace_to_inter_racial_lastnames"
    ):
        valid_names = (
            'white_names',
            'black_names',
            'hispanic_names',
            'asian_names',
            'native_american_names',
            'inter_racial_names'
        )

        for data_dict in custom_data:
            if 'name' not in data_dict:
                raise ValueError("Invalid JSON format. 'name' key is missing.")

            name = data_dict['name']
            first_names = data_dict.get('first_names', [])
            last_names = data_dict.get('last_names', [])

            if not isinstance(name, str):
                raise ValueError("Invalid 'name' format in the JSON file.")

            if name not in valid_names:
                raise ValueError(f"Invalid 'name' value '{name}'. It should be one of: {', '.join(valid_names)}.")

            if not first_names and not last_names:
                raise ValueError(f"At least one of 'first_names' or 'last_names' must be specified for '{name}'.")

            if set(data_dict.keys()) - {'name', 'first_names', 'last_names'}:
                raise ValueError(f"Invalid keys in the JSON for '{name}'. "
                                    f"Only the following keys are allowed: 'name', 'first_names', 'last_names'.")

            if name == 'white_names':
                white_names['first_names'] += first_names
                white_names['last_names'] += last_names
            elif name == 'black_names':
                black_names['first_names'] += first_names
                black_names['last_names'] += last_names
            elif name == 'hispanic_names':
                hispanic_names['first_names'] += first_names
                hispanic_names['last_names'] += last_names
            elif name == 'asian_names':
                asian_names['first_names'] += first_names
                asian_names['last_names'] += last_names
            elif name == 'native_american_names':
                native_american_names['last_names'] += last_names
            elif name == 'inter_racial_names':
                inter_racial_names['last_names'] += last_names

    elif new_name in (
            "replace_to_male_pronouns",
            "replace_to_female_pronouns",
            "replace_to_neutral_pronouns"
    ):
        valid_names = ('female_pronouns', 'male_pronouns', 'neutral_pronouns')

        for data_dict in custom_data:
            if 'name' not in data_dict:
                raise ValueError("Invalid JSON format. 'name' key is missing.")

            name = data_dict['name']

            if name not in valid_names:
                raise ValueError(
                    f"Invalid 'name' value '{name}'. It should be one of: {', '.join(valid_names)}.")

            pronouns = {
                'subjective_pronouns': data_dict.get('subjective_pronouns', []),
                'objective_pronouns': data_dict.get('objective_pronouns', []),
                'reflexive_pronouns': data_dict.get('reflexive_pronouns', []),
                'possessive_pronouns': data_dict.get('possessive_pronouns', [])
            }

            if all(key not in pronouns for key in
                    ['subjective_pronouns', 'objective_pronouns', 'reflexive_pronouns', 'possessive_pronouns']):
                raise ValueError(
                    f"Missing pronoun keys in the JSON for '{name}'. Please include at least one of: "
                    "'subjective_pronouns', 'objective_pronouns', 'reflexive_pronouns', 'possessive_pronouns'.")

            invalid_keys = set(data_dict.keys()) - {'name', 'subjective_pronouns', 'objective_pronouns',
                                                        'reflexive_pronouns', 'possessive_pronouns'}
            if invalid_keys:
                raise ValueError(
                    f"Invalid keys in the JSON for '{name}': {', '.join(invalid_keys)}. "
                    f"Only the following keys are allowed: "
                    "'name', 'subjective_pronouns', 'objective_pronouns', 'reflexive_pronouns', 'possessive_pronouns'.")

            if name == 'female_pronouns':
                female_pronouns['subjective_pronouns'] += pronouns['subjective_pronouns']
                female_pronouns['objective_pronouns'] += pronouns['objective_pronouns']
                female_pronouns['reflexive_pronouns'] += pronouns['reflexive_pronouns']
                female_pronouns['possessive_pronouns'] += pronouns['possessive_pronouns']
            elif name == 'male_pronouns':
                male_pronouns['subjective_pronouns'] += pronouns['subjective_pronouns']
                male_pronouns['objective_pronouns'] += pronouns['objective_pronouns']
                male_pronouns['reflexive_pronouns'] += pronouns['reflexive_pronouns']
                male_pronouns['possessive_pronouns'] += pronouns['possessive_pronouns']
            elif name == 'neutral_pronouns':
                neutral_pronouns['subjective_pronouns'] += pronouns['subjective_pronouns']
                neutral_pronouns['objective_pronouns'] += pronouns['objective_pronouns']
                neutral_pronouns['reflexive_pronouns'] += pronouns['reflexive_pronouns']
                neutral_pronouns['possessive_pronouns'] += pronouns['possessive_pronouns']

