import asyncio
import pytest
from langtest.transform.representation import *
from langtest.utils.custom_types.sample import *


class TestRepresentation:
    """
    A test suite for evaluating the transformation process of various representations.

    This test suite ensures that the representations can successfully transform input data
    and produce valid results.

    The representations tested include GenderRepresentation, EthnicityRepresentation,
    ReligionRepresentation, and CountryEconomicRepresentation.

    Attributes:
        None

    """

    @pytest.fixture
    def sample_data(self):
        """
        A fixture providing sample data for the representation transformation tests.

        Returns:
            list: A list containing sample SequenceClassificationSample instances.

        """
        return [
            SequenceClassificationSample(
                original="The last good ernest movie, and the best at that. how can you not laugh at least once at this movie. the last line is a classic, as is ernest's gangster impressions, his best moment on film. this has his best lines and is a crowning achievement among the brainless screwball comedies."
            ),
            SequenceClassificationSample(
                original="After my 6 year old daughter began taking riding lessons I started looking for horse movies for her. I had always heard of National Velvet but had never seen it. Boy am I glad I bought it! It's become a favorite of mine, my 6 year old AND my 2 year old. It's a shame movies like this aren't made anymore."
            ),
            SequenceClassificationSample(
                original="I had no expectations when seeing the movie because i was seeing it with a bunch of friends and had no idea what it was. some parts were silly and some parts were lame, but overall the movie was worth watching. i like goth looking women; this movie has plenty of it. the fangs do look really lame though."
            ),
            SequenceClassificationSample(
                original="I picked up this dvd for $4.99. they had put spiffy cover art on the package, along with a plot summary that had nothing to do with the movie. the acting is terrible, and the writing is worse. the only possible way this movie could be redeemed would be as mst3k fodder. i paid too much."
            ),
            SequenceClassificationSample(
                original="This movie moves and inspire you, it's like you are one of the family. just to see and witness life during the depression era, makes you feel humble and grateful. jonathan silverman delivered well, so convincing and very witty! a must see for teens!"
            ),
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
        """
        Test case for representation classes.

        Args:
            representation (Representation): The representation class to be tested.
            sample_data (list): A list containing sample SequenceClassificationSample instances.

        Returns:
            None

        Raises:
            AssertionError: If the transformation or the final result is invalid.

        """
        # Define the representation configurations
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

        # Call the representation's transform method with the representation type and parameters
        transform_result = representation.transform(
            representation, sample_data, representation_config
        )

        # Ensure the transform result is a list
        assert isinstance(transform_result, list)

        # Run the transform asynchronously using the representation class and sample data
        result = asyncio.run(
            representation.run(
                transform_result, model="lvwerra/distilbert-imdb", raw_data=sample_data
            )
        )

        # Ensure the final result is a list
        assert isinstance(result, list)

        # Ensure each result in the list has a non-null value for the `actual_results` attribute
        for r in result:
            assert r.actual_results is not None
