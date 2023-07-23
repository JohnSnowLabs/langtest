import unittest
from langtest.transform.representation import *
from langtest.utils.custom_types.sample import *


class RepresentationTestCase(unittest.TestCase):
    """
    Test case for representation classes.

    """

    def setUp(self):
        """
        Set up the test case.
        """
        # Create sample data
        self.sentences = [
            SequenceClassificationSample(
                original="Just as a reminder to anyone just now reading the comments on this excellent BBC mini-series, published in 1981, it was not available on DVD until the last few years. Since then, it has become available, but initially only in the British format (for which I bought an 'international' DVD player, which you have to hack--illegally, I suspect, to see it), but the series is now available through amazon.com--3 discs-- for between $19-21, to be viewed on DVD in the US format, no hacking. There were 41 reviews, average 5 stars. This mini-series is one of the very best on Oppenheimer, or the Manhattan Project, or virtually anything produced by the BBC."
            )
        ]

        # Define the gender representation configurations
        self.gender_representation = {
            "min_gender_representation_count": {"min_count": 5},
            "min_gender_representation_proportion": {"min_proportion": 0.1},
        }

        self.ethnicity_representation = {
            "min_ethnicity_name_representation_count": {"min_count": 10},
            "min_ethnicity_name_representation_proportion": {"min_proportion": 0.1},
        }

        self.religion_representation = {
            "min_religion_name_representation_count": {"min_count": 10},
            "min_religion_name_representation_proportion": {"min_proportion": 0.1},
        }

        self.country_representation = {
            "min_country_economic_representation_count": {"min_count": 10},
            "min_country_economic_representation_proportion": {"min_proportion": 0.1},
        }

    async def run_transform(self, representation_class, transform_result):
        """Run the transform asynchronously.

        Parameters:
        - representation_class (type): The representation class to be used.
        - transform_result (list): The result of the transform operation.

        Returns:
        - list: The transformed result.

        Raises:
        - Exception: If an error occurs during the asynchronous execution.
        """
        result = await representation_class.run(
            transform_result, model="lvwerra/distilbert-imdb", raw_data=self.sentences
        )
        return result

    def test_gender_representation(self):
        """
        Test the gender representation.
        """
        for rep_type, params in self.gender_representation.items():
            with self.subTest(rep_type=rep_type):
                transform_result = GenderRepresentation.transform(
                    rep_type, self.sentences, params
                )
                self.assertIsInstance(transform_result, list)
                final_result = asyncio.run(
                    self.run_transform(GenderRepresentation, transform_result)
                )
                self.assertIsInstance(final_result, list)
                for result in final_result:
                    self.assertNotEqual(result.actual_results, None)

    def test_ethnicity_representation(self):
        """
        Test the ethnicity representation.
        """

        for rep_type, params in self.ethnicity_representation.items():
            with self.subTest(rep_type=rep_type):
                transform_result = EthnicityRepresentation.transform(
                    rep_type, self.sentences, params
                )
                self.assertIsInstance(transform_result, list)
                final_result = asyncio.run(
                    self.run_transform(EthnicityRepresentation, transform_result)
                )
                self.assertIsInstance(final_result, list)
                for result in final_result:
                    self.assertNotEqual(result.actual_results, None)

    def test_religion_representation(self):
        """
        Test the religion representation.
        """

        for rep_type, params in self.religion_representation.items():
            with self.subTest(rep_type=rep_type):
                transform_result = ReligionRepresentation.transform(
                    rep_type, self.sentences, params
                )
                self.assertIsInstance(transform_result, list)
                final_result = asyncio.run(
                    self.run_transform(ReligionRepresentation, transform_result)
                )
                self.assertIsInstance(final_result, list)
                for result in final_result:
                    self.assertNotEqual(result.actual_results, None)

    def test_country_representation(self):
        """
        Test the country representation.
        """

        for rep_type, params in self.country_representation.items():
            with self.subTest(rep_type=rep_type):
                transform_result = CountryEconomicRepresentation.transform(
                    rep_type, self.sentences, params
                )
                self.assertIsInstance(transform_result, list)
                final_result = asyncio.run(
                    self.run_transform(CountryEconomicRepresentation, transform_result)
                )
                self.assertIsInstance(final_result, list)
                for result in final_result:
                    self.assertNotEqual(result.actual_results, None)
