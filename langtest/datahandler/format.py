from abc import ABC, abstractmethod
from typing import Tuple, List, Union
import re
from ..utils.custom_types import Sample


class BaseFormatter(ABC):
    """Abstract base class for defining formatter classes.

    Subclasses should implement the static methods `to_csv` and `to_conll`.
    """

    @staticmethod
    @abstractmethod
    def to_csv(sample: Sample):
        """Converts a custom type to a CSV string.

        Args:
            sample (Sample): The custom type to convert.

        Returns:
            The CSV string representation of the custom type.

        Raises:
            NotImplementedError: This method should be implemented by the subclass.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def to_conll(sample: Sample):
        """Converts a custom type to a CoNLL string.

        Args:
            sample (Sample): The custom type to convert.

        Returns:
            The CoNLL string representation of the custom type.

        Raises:
            NotImplementedError: This method should be implemented by the subclass.
        """
        raise NotImplementedError()


class Formatter:
    """Formatter class for converting between custom types and different output formats.

    This class uses the `to_csv` and `to_conll` methods of subclasses of `BaseFormatter`
    to perform the conversions. The appropriate subclass is selected based on the
    type of the expected results in the `sample` argument.
    """

    @staticmethod
    def process(sample: Sample, output_format: str, *args, **kwargs):
        """Method to format the sample into the desired format

        Args:
            sample (Sample):
                The input sample to convert.
            output_format (str):
                The output format to convert to, either "csv" or "conll".
            *args:
                Optional positional arguments to pass to the `to_csv` or `to_conll` methods.
            **kwargs:
                Optional keyword arguments to pass to the `to_csv` or `to_conll` methods.

        Returns:
            The output string in the specified format.

        Raises:
            NameError: If no formatter subclass is defined for the type of the expected results in the sample.

        """
        formats = {cls.__name__: cls for cls in BaseFormatter.__subclasses__()}
        class_name = type(sample.expected_results).__name__
        try:
            return getattr(formats[f"{class_name}Formatter"], f"to_{output_format}")(
                sample, *args, **kwargs
            )
        except KeyError:
            raise NameError(f"Class '{class_name}Formatter' not yet implemented.")


class SequenceClassificationOutputFormatter(BaseFormatter, ABC):
    """Formatter class for converting `SequenceClassificationOutput` objects to CSV.

    The `to_csv` method returns a CSV string representing the `SequenceClassificationOutput`
    object in the sample argument.
    """

    @staticmethod
    def to_csv(sample: Sample) -> str:
        """Convert a Sample object into a row for exporting.

        Args:
            Sample:
                Sample object to convert.

        Returns:
            List[str]:
                Row formatted as a list of strings.
        """
        if sample.test_case:
            row = [sample.test_case, sample.expected_results.predictions[0].label]
        else:
            row = [sample.original, sample.expected_results.predictions[0].label]
        return row


class NEROutputFormatter(BaseFormatter):
    """Formatter class for converting `NEROutput` objects to CSV and CoNLL.

    The `to_csv` method returns a CSV string representing the `NEROutput` object in the sample
    argument. The `to_conll` method returns a CoNLL string representing the `NEROutput` object.
    """

    @staticmethod
    def to_csv(
        sample: Sample, delimiter: str = ","
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Converts a custom type to a CSV string.

        Args:
            sample (Sample):
                The input sample containing the `NEROutput` object to convert.
            delimiter (str):
                The delimiter character to use in the CSV string.

        Returns:
            Tuple[List[str], List[str], List[str], List[str]]:
                tuple containing the list of tokens of the original sentence, the list of
                labels of the original sentence, the list of tokens for the perturbed sentence
                and the labels of the perturbed sentence.
        """
        test_case = sample.test_case
        original = sample.original

        words = re.finditer(r"([^\s]+)", original)
        tokens, labels = [], []

        for word in words:
            tokens.append(word.group())
            match = sample.expected_results[word.group()]
            labels.append(match.entity if match is not None else "O")

        if test_case and sample.actual_results:
            test_case_words = re.finditer(r"([^\s]+)", test_case)
            test_case_tokens, test_case_labels = [], []

            for word in test_case_words:
                test_case_tokens.append(word.group())
                match = sample.actual_results[word.group()]
                test_case_labels.append(match.entity if match is not None else "O")

            assert len([token for token in test_case_tokens if token != "O"]) == len(
                sample.actual_results
            )
            return tokens, labels, test_case_tokens, test_case_labels
        return tokens, labels, [], []

    @staticmethod
    def to_conll(
        sample: Sample, writing_mode: str = "ignore"
    ) -> Union[str, Tuple[str, str]]:
        """Converts a custom type to a CoNLL string.

        Args:
            sample (Sample):
                The input sample containing the `NEROutput` object to convert.
            writing_mode (str):
                what to do with the expected results if present:
                - ignore: simply ignores the expected_results
                - append: the formatted expected_results to the original ones
                - separate: returns a formatted string for the original sentence and one for
                            the perturbed sentence

        Returns:
            The CoNLL string representation of the custom type.
        """
        assert writing_mode in [
            "ignore",
            "append",
            "separate",
        ], f"writing_mode: {writing_mode} not supported."

        text, text_perturbed = "", ""
        test_case = sample.test_case
        original = sample.original

        words = re.finditer(r"([^\s]+)", original)

        for word in words:
            token = word.group()
            match = sample.expected_results[word.group()]
            label = match.entity if match is not None else "O"
            text += f"{token} -X- -X- {label}\n"

        if test_case and writing_mode != "ignore":
            words = re.finditer(r"([^\s]+)", test_case)

            for word in words:
                token = word.group()
                match = sample.actual_results[word.group()]
                label = match.entity if match is not None else "O"
                if writing_mode == "append":
                    text += f"{token} -X- -X- {label}\n"
                elif writing_mode == "separate":
                    text_perturbed += f"{token} -X- -X- {label}\n"

        if writing_mode == "separate":
            return text, text_perturbed
        return text
