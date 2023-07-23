import asyncio
import pytest
from langtest.transform.representation import *
from langtest.utils.custom_types.sample import *


class TestRepresentationTransform:
    @pytest.fixture
    def sample_data(self):
        return [
            SequenceClassificationSample(
                original="Just as a reminder to anyone just now reading the comments on this excellent BBC mini-series, published in 1981, it was not available on DVD until the last few years. Since then, it has become available, but initially only in the British format (for which I bought an 'international' DVD player, which you have to hack--illegally, I suspect, to see it), but the series is now available through amazon.com--3 discs-- for between $19-21, to be viewed on DVD in the US format, no hacking. There were 41 reviews, average 5 stars. This mini-series is one of the very best on Oppenheimer, or the Manhattan Project, or virtually anything produced by the BBC."
            )
        ]

    @pytest.mark.parametrize(
        "representation",
        [
            GenderRepresentation,
            EthnicityRepresentation,
            ReligionRepresentation,
            CountryEconomicRepresentation,
        ],
    )
    def test_transform(self, representation, sample_data):

        representation_config = {
            "min_gender_representation_count": {"min_count": 5},
            "min_gender_representation_proportion": {"min_proportion": 0.1},
            "min_ethnicity_name_representation_count": {"min_count": 10},
            "min_ethnicity_name_representation_proportion": {"min_proportion": 0.1},
            "min_religion_name_representation_count": {"min_count": 10},
            "min_religion_name_representation_proportion": {"min_proportion": 0.1},
            "min_country_economic_representation_count": {"min_count": 10},
            "min_country_economic_representation_proportion": {"min_proportion": 0.1},
        }

        transform_result = representation.transform(
            representation, sample_data, representation_config
        )
        assert isinstance(transform_result, list)

        result = asyncio.run(
            representation.run(
                transform_result, model="lvwerra/distilbert-imdb", raw_data=sample_data
            )
        )
        assert isinstance(result, list)
        for r in result:
            assert r.actual_results is not None
