from abc import ABC, abstractmethod
from typing import Tuple

from ..utils.custom_types import Sample


class BaseFormatter(ABC):
    """
    Abstract base class for defining formatter classes.
    Subclasses should implement the static methods `to_csv` and `to_conll`.
    """

    @staticmethod
    @abstractmethod
    def to_csv(custom_type):
        """
        Converts a custom type to a CSV string.

        Args:
            custom_type: The custom type to convert.

        Returns:
            The CSV string representation of the custom type.

        Raises:
            NotImplementedError: This method should be implemented by the subclass.
        """

        return NotImplementedError

    @staticmethod
    @abstractmethod
    def to_conll(custom_type):
        """
        Converts a custom type to a CoNLL string.

        Args:
            custom_type: The custom type to convert.

        Returns:
            The CoNLL string representation of the custom type.

        Raises:
            NotImplementedError: This method should be implemented by the subclass.
        """

        return NotImplementedError


class Formatter:
    """
    Formatter class for converting between custom types and different output formats.

    This class uses the `to_csv` and `to_conll` methods of subclasses of `BaseFormatter`
    to perform the conversions. The appropriate subclass is selected based on the
    type of the expected results in the `sample` argument.
    """

    @staticmethod
    def process(sample: Sample, output_format: str, *args, **kwargs):
        """
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
            return getattr(formats[f"{class_name}Formatter"], f"to_{output_format}")(sample, *args, **kwargs)
        except KeyError:
            raise NameError(
                f"Class '{class_name}Formatter' not yet implemented.")


class SequenceClassificationOutputFormatter(BaseFormatter, ABC):
    """
    Formatter class for converting `SequenceClassificationOutput` objects to CSV.

    The `to_csv` method returns a CSV string representing the `SequenceClassificationOutput`
    object in the sample argument.
    """

    @staticmethod
    def to_csv(sample: Sample, delimiter: str = ",") -> str:
        """
        Args:
            sample (Sample):
                The input sample containing the `SequenceClassificationOutput` object to convert.
            delimiter (str):
                The delimiter character to use in the CSV string.

        Returns:
            str: The CSV string representation of the `SequenceClassificationOutput` object.
        """
        original = sample.original
        test_case = sample.test_case
        if test_case:
            return f"{test_case}{delimiter}{sample.expected_results.to_str_list()[0]}\n"
        else:
            return f"{original}{delimiter}{sample.expected_results.to_str_list()[0]}\n"


class NEROutputFormatter(BaseFormatter):
    """
    Formatter class for converting `NEROutput` objects to CSV and CoNLL.

    The `to_csv` method returns a CSV string representing the `NEROutput` object in the sample
    argument. The `to_conll` method returns a CoNLL string representing the `NEROutput` object.
    """

    @staticmethod
    def to_csv(sample: Sample, delimiter: str = ",", temp_id: int = None) -> Tuple[str, int]:
        """
        Args:
            sample (Sample):
                The input sample containing the `NEROutput` object to convert.
            delimiter (str):
                The delimiter character to use in the CSV string.
            temp_id (int):
                A temporary ID to use for grouping entities by document.

        Returns:
            Tuple[str, int]:
                The CSV or CoNLL string representation of the `NEROutput` object along with the document id
        """
        text = ""
        test_case = sample.test_case
        original = sample.original
        if test_case:
            test_case_items = test_case.split()
            norm_test_case_items = test_case.lower().split()
            norm_original_items = original.lower().split()
            temp_len = 0
            for jdx, item in enumerate(norm_test_case_items):
                if item in norm_original_items and jdx >= norm_original_items.index(item):
                    oitem_index = norm_original_items.index(item)
                    j = sample.expected_results.predictions[oitem_index + temp_len]
                    if temp_id != j.doc_id and jdx == 0:
                        text += f"{j.doc_name}\n\n"
                        temp_id = j.doc_id
                    text += f"{test_case_items[jdx]}{delimiter}{j.pos_tag}{delimiter}{j.chunk_tag}{delimiter}{j.entity}\n"
                    norm_original_items.pop(oitem_index)
                    temp_len += 1
                else:
                    o_item = norm_original_items[jdx - temp_len]
                    letters_count = len(set(o_item) - set(item))
                    if len(norm_test_case_items) == len(norm_original_items) or letters_count < len(o_item):
                        tl = sample.expected_results.predictions[jdx]
                        text += f"{test_case_items[jdx]}{delimiter}{tl.pos_tag}{delimiter}{tl.chunk_tag}{delimiter}{tl.entity}\n"
                    else:
                        text += f"{test_case_items[jdx]}{delimiter}O{delimiter}O{delimiter}O\n"
            text += "\n"

        else:
            for j in sample.expected_results.predictions:
                if temp_id != j.doc_id:
                    text += f"{j.doc_name}\n\n"
                    temp_id = j.doc_id
                text += f"{j.span.word}{delimiter}{j.pos_tag}{delimiter}{j.chunk_tag}{delimiter}{j.entity}\n"
            text += "\n"
        return text, temp_id

    @staticmethod
    def to_conll(sample: Sample, temp_id: int = None) -> Tuple[str, int]:
        """
        Args:
            sample (Sample):
                The input sample containing the `NEROutput` object to convert.
            temp_id (int):
                A temporary ID to use for grouping entities by document.

        Returns:
            The CoNLL string representation of the custom type.
        """
        text = ""
        test_case = sample.test_case
        original = sample.original
        if test_case:
            test_case_items = test_case.split()
            norm_test_case_items = test_case.lower().split()
            norm_original_items = original.lower().split()
            temp_len = 0
            for jdx, item in enumerate(norm_test_case_items):
                try:
                    if item in norm_original_items and jdx >= norm_original_items.index(item):
                        oitem_index = norm_original_items.index(item)
                        j = sample.expected_results.predictions[oitem_index + temp_len]
                        if temp_id != j.doc_id and jdx == 0:
                            text += f"{j.doc_name}\n\n"
                            temp_id = j.doc_id
                        text += f"{test_case_items[jdx]} {j.pos_tag} {j.chunk_tag} {j.entity}\n"
                        norm_original_items.pop(oitem_index)
                        temp_len += 1
                    else:
                        o_item = sample.expected_results.predictions[jdx].span.word
                        letters_count = len(set(item) - set(o_item))
                        if len(norm_test_case_items) == len(original.lower().split()) or letters_count < 2:
                            tl = sample.expected_results.predictions[jdx]
                            text += f"{test_case_items[jdx]} {tl.pos_tag} {tl.chunk_tag} {tl.entity}\n"
                        else:
                            text += f"{test_case_items[jdx]} O O O\n"
                except IndexError:
                    text += f"{test_case_items[jdx]} O O O\n"
            text += "\n"

        else:
            for j in sample.expected_results.predictions:
                if temp_id != j.doc_id:
                    text += f"{j.doc_name}\n\n"
                    temp_id = j.doc_id
                text += f"{j.span.word} {j.pos_tag} {j.chunk_tag} {j.entity}\n"
            text += "\n"
        return text, temp_id
