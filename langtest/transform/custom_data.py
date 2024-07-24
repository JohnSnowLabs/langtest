from .constants import (
    asian_names,
    black_names,
    country_economic_dict,
    hispanic_names,
    inter_racial_names,
    male_pronouns,
    native_american_names,
    neutral_pronouns,
    religion_wise_names,
    white_names,
    female_pronouns,
)
from typing import Union
from ..errors import Errors


def add_custom_data(data: Union[list, dict], name: str, append: bool) -> None:
    """
    Adds custom data to the corresponding bias dictionaries based on the specified name.

    Args:
        data (dict or list): The data to be added.
        name (str): The type of dictionary to update. It should be one of the following:
            - "Country-Economic-Bias"
            - "Religion-Bias"
            - "Ethnicity-Name-Bias"
            - "Gender-Pronoun-Bias"
            - "Country-Economic-Representation"
            - "Religion-Representation"
            - "Ethnicity-Representation"
        append (bool): Specifies whether to append the values or overwrite them.

    Raises:
        ValueError: If the specified `name` is invalid or if the provided data has an invalid format or contains invalid keys.
    """
    if name in ("Country-Economic-Bias", "Country-Economic-Representation"):
        valid_names = country_economic_dict.keys()

        # Validate the schema
        if not set(data.keys()).issubset(valid_names):
            raise ValueError(Errors.E054(var=", ".join(valid_names)))

        if append:
            # Append unique values to existing keys
            for key, values in data.items():
                unique_values = set(values)
                country_economic_dict[key] = list(
                    set(country_economic_dict[key]) | unique_values
                )
        else:
            # Overwrite the keys' values
            for key, values in data.items():
                country_economic_dict[key] = values

    elif name in ("Religion-Bias", "Religion-Representation"):
        valid_names = religion_wise_names.keys()

        # Validate the schema
        if not set(data.keys()).issubset(valid_names):
            raise ValueError(Errors.E054(var=", ".join(valid_names)))

        if append:
            # Append unique values to existing keys
            for key, values in data.items():
                unique_values = set(values)
                religion_wise_names[key] = list(
                    set(religion_wise_names[key]) | unique_values
                )
        else:
            # Overwrite the keys' values
            for key, values in data.items():
                religion_wise_names[key] = values

    elif name in ("Ethnicity-Name-Bias", "Ethnicity-Representation"):
        ethnicity_data = {
            "white_names": white_names,
            "black_names": black_names,
            "hispanic_names": hispanic_names,
            "asian_names": asian_names,
            "native_american_names": native_american_names,
            "inter_racial_names": inter_racial_names,
        }

        valid_names = tuple(ethnicity_data.keys())

        for data_dict in data:
            if "name" not in data_dict:
                raise ValueError(Errors.E055())

            name = data_dict["name"]
            first_names = data_dict.get("first_names", [])
            last_names = data_dict.get("last_names", [])

            if not isinstance(name, str):
                raise ValueError(Errors.E057())

            if name not in valid_names:
                raise ValueError(Errors.E056(var1=name, var2=", ".join(valid_names)))

            if not first_names and not last_names:
                if name not in ("native_american_names", "inter_racial_names"):
                    raise ValueError(Errors.E058(name=name))
                else:
                    raise ValueError(Errors.E059(name=name))

            if set(data_dict.keys()) - {"name", "first_names", "last_names"}:
                raise ValueError(Errors.E060(name=name))

            if name in (
                "white_names",
                "black_names",
                "hispanic_names",
                "asian_names",
            ):
                if append:
                    ethnicity_data[name]["first_names"] += first_names
                    ethnicity_data[name]["last_names"] += last_names
                else:
                    ethnicity_data[name]["first_names"] = first_names
                    ethnicity_data[name]["last_names"] = last_names
            elif name in ("native_american_names", "inter_racial_names"):
                if append:
                    ethnicity_data[name]["last_names"] += last_names
                else:
                    ethnicity_data[name]["last_names"] = last_names

    elif name == "Gender-Pronoun-Bias":
        valid_names = ("female_pronouns", "male_pronouns", "neutral_pronouns")

        # Validate the schema
        for data_dict in data:
            if "name" not in data_dict:
                raise ValueError(Errors.E055())

            name = data_dict["name"]

            if name not in valid_names:
                raise ValueError(Errors.E056(var1=name, var2=", ".join(valid_names)))

            pronouns = {
                "subjective_pronouns": data_dict.get("subjective_pronouns", []),
                "objective_pronouns": data_dict.get("objective_pronouns", []),
                "reflexive_pronouns": data_dict.get("reflexive_pronouns", []),
                "possessive_pronouns": data_dict.get("possessive_pronouns", []),
            }

            if all(
                key not in pronouns
                for key in [
                    "subjective_pronouns",
                    "objective_pronouns",
                    "reflexive_pronouns",
                    "possessive_pronouns",
                ]
            ):
                raise ValueError(Errors.E061(name=name))

            invalid_keys = set(data_dict.keys()) - {
                "name",
                "subjective_pronouns",
                "objective_pronouns",
                "reflexive_pronouns",
                "possessive_pronouns",
            }
            if invalid_keys:
                raise ValueError(Errors.E062(var1=name, var2=", ".join(invalid_keys)))

            bias_dict = {
                "female_pronouns": female_pronouns,
                "male_pronouns": male_pronouns,
                "neutral_pronouns": neutral_pronouns,
            }

            if name in bias_dict:
                bias = bias_dict[name]
                if append:
                    bias["subjective_pronouns"].extend(
                        set(pronouns["subjective_pronouns"])
                        - set(bias["subjective_pronouns"])
                    )
                    bias["objective_pronouns"].extend(
                        set(pronouns["objective_pronouns"])
                        - set(bias["objective_pronouns"])
                    )
                    bias["reflexive_pronouns"].extend(
                        set(pronouns["reflexive_pronouns"])
                        - set(bias["reflexive_pronouns"])
                    )
                    bias["possessive_pronouns"].extend(
                        set(pronouns["possessive_pronouns"])
                        - set(bias["possessive_pronouns"])
                    )
                else:
                    bias["subjective_pronouns"] = pronouns["subjective_pronouns"]
                    bias["objective_pronouns"] = pronouns["objective_pronouns"]
                    bias["reflexive_pronouns"] = pronouns["reflexive_pronouns"]
                    bias["possessive_pronouns"] = pronouns["possessive_pronouns"]
