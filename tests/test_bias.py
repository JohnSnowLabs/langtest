import unittest
from langtest.utils.custom_types import SequenceClassificationSample, Transformation, Span
from langtest.transform.bias import (
    GenderPronounBias,
    CountryEconomicBias,
    EthnicityNameBias,
    ReligionBias,
)


class TestBias(unittest.TestCase):
    """
    Test suite for bias transformation functions.

    Note: we are using `SequenceClassificationSample` but any `xxxSample` would work
    """

    def test_gender_bias(self):
        """
        Test gender bias transformation.
        """
        sample = SequenceClassificationSample(
            original="Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday .",
            test_type="replace_to_female_pronouns",
        )

        sample = GenderPronounBias.transform(
            sample_list=[sample], pronouns_to_substitute=["their"], pronoun_type="female"
        )[0]

        assert len(sample.transformations) == 1
        assert sample.transformations[0] == Transformation(
            original_span=Span(start=27, end=32, word="their"),
            new_span=Span(start=27, end=31, word="hers"),
        ) or sample.transformations[0] == Transformation(
            original_span=Span(start=27, end=32, word="their"),
            new_span=Span(start=27, end=30, word="her"),
        )

        assert (
            sample.test_case
            == "Japan began the defence of hers Asian Cup title with a lucky 2-1 win against Syria "
            "in a Group C championship match on Friday ."
            or sample.test_case
            == "Japan began the defence of her Asian Cup title with a lucky 2-1 win against Syria in "
            "a Group C championship match on Friday ."
        )

    def test_country_economic_bias(self):
        """
        Test country economic bias transformation.
        """
        sample = SequenceClassificationSample(
            original="Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria.",
            test_type="replace_to_low_income_country",
        )
        sample = CountryEconomicBias.transform(
            sample_list=[sample],
            country_names_to_substitute=["Japan"],
            chosen_country_names=["Nigeria"],
        )[0]

        assert len(sample.transformations) == 1
        assert sample.transformations[0] == Transformation(
            original_span=Span(start=0, end=5, word="Japan"),
            new_span=Span(start=0, end=7, word="Nigeria"),
        )

    def test_ethnicity_bias(self):
        """
        Test ethnicity bias transformation.
        """
        sample = SequenceClassificationSample(
            original="He was born in the USA and was called Malcolm after his grandfather",
            test_type="replace_to_white_firstnames",
        )
        sample = EthnicityNameBias.transform(
            sample_list=[sample],
            names_to_substitute=["Malcolm"],
            chosen_ethnicity_names=["James"],
        )[0]

        assert len(sample.transformations) == 1
        assert sample.transformations[0] == Transformation(
            original_span=Span(start=38, end=45, word="Malcolm"),
            new_span=Span(start=38, end=43, word="James"),
        )

    def test_religion_bias(self):
        """
        Test religion bias transformation.
        """
        sample = SequenceClassificationSample(
            original="He was born in the USA and was called Malcolm after his grandfather",
            test_type="replace_to_hindu_names",
        )
        sample = ReligionBias.transform(
            sample_list=[sample], names_to_substitute=["Malcolm"], chosen_names=["Ankit"]
        )[0]

        assert len(sample.transformations) == 1
        assert sample.transformations[0] == Transformation(
            original_span=Span(start=38, end=45, word="Malcolm"),
            new_span=Span(start=38, end=43, word="Ankit"),
        )
