import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from ..utils.custom_types import NERSample, Sample, SequenceClassificationSample, QASample
from ..errors import Errors


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
            if sample.task == "question-answering":
                return getattr(QAFormatter, f"to_{output_format}")(
                    sample, *args, **kwargs
                )

            return getattr(formats[f"{class_name}Formatter"], f"to_{output_format}")(
                sample, *args, **kwargs
            )
        except KeyError:
            raise NameError(Errors.E031(class_name=class_name))


class SequenceClassificationOutputFormatter(BaseFormatter, ABC):
    """Formatter class for converting `SequenceClassificationOutput` objects to CSV.

    The `to_csv` method returns a CSV string representing the `SequenceClassificationOutput`
    object in the sample argument.
    """

    @staticmethod
    def to_csv(sample: SequenceClassificationSample) -> Tuple[str, str]:
        """Convert a SequenceClassificationSample object into a row for exporting.

        Args:
            sample (SequenceClassificationSample) : Sample object to convert.

        Returns:
            Tuple[str, str]:
                Row formatted as a list of strings.
        """
        if sample.test_case:
            return [sample.test_case, sample.expected_results.predictions[0].label]
        return [sample.original, sample.expected_results.predictions[0].label]


class NEROutputFormatter(BaseFormatter):
    """Formatter class for converting `NEROutput` objects to CSV and CoNLL.

    The `to_csv` method returns a CSV string representing the `NEROutput` object in the sample
    argument. The `to_conll` method returns a CoNLL string representing the `NEROutput` object.
    """

    @staticmethod
    def to_csv(
        sample: NERSample, delimiter: str = ",", temp_id: int = None
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Converts a NERSample to a CSV string.

        Args:
            sample (NERSample):
                The input sample containing the `NEROutput` object to convert.
            delimiter (str):
                The delimiter character to use in the CSV string.
            temp_id (int):
                A temporary ID to use for grouping entities by document.

        Returns:
            Tuple[List[str], List[str], List[str], List[str]]:
                The CSV or CoNLL string representation of the `NEROutput` object along with the document id
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
    def to_conll(sample: NERSample, temp_id: int = None) -> Union[str, Tuple[str, str]]:
        """Converts a custom type to a CoNLL string.

        Args:
            sample (NERSample):
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

        text = ""
        test_case = sample.test_case
        original = sample.original
        if test_case:
            test_case_items = test_case.split(" ")
            norm_test_case_items = test_case.lower().split(" ")
            norm_original_items = original.lower().split(" ")
            temp_len = 0
            for jdx, item in enumerate(norm_test_case_items):
                if test_case_items[jdx] == "\n":
                    text += "\n"  # add a newline character after each sentence
                else:
                    try:
                        if (
                            item in norm_original_items
                            and jdx >= norm_original_items.index(item)
                        ):
                            oitem_index = norm_original_items.index(item)
                            j = sample.expected_results.predictions[
                                oitem_index + temp_len
                            ]
                            if temp_id != j.doc_id and jdx == 0:
                                text += f"{j.doc_name}\n\n"
                                temp_id = j.doc_id
                            text += f"{test_case_items[jdx]} {j.pos_tag} {j.chunk_tag} {j.entity}\n"
                            norm_original_items.pop(oitem_index)
                            temp_len += 1
                        else:
                            o_item = sample.expected_results.predictions[jdx].span.word
                            letters_count = len(set(item) - set(o_item))
                            if (
                                len(norm_test_case_items)
                                == len(original.lower().split(" "))
                                or letters_count < 2
                            ):
                                tl = sample.expected_results.predictions[jdx]
                                text += f"{test_case_items[jdx]} {tl.pos_tag} {tl.chunk_tag} {tl.entity}\n"
                            else:
                                text += f"{test_case_items[jdx]} -X- -X- O\n"
                    except IndexError:
                        text += f"{test_case_items[jdx]} -X- -X- O\n"

        else:
            for j in sample.expected_results.predictions:
                if temp_id != j.doc_id:
                    text += f"{j.doc_name}\n\n"
                    temp_id = j.doc_id
                text += f"{j.span.word} {j.pos_tag} {j.chunk_tag} {j.entity}\n"

        return text, temp_id


class QAFormatter(BaseFormatter):
    def to_jsonl(sample: QASample, *args, **kwargs):
        """Converts a QASample to a JSONL string."""

        context = sample.original_context
        question = sample.original_question
        options = sample.options

        # override if perturbed values are present
        if sample.perturbed_context:
            context = sample.perturbed_context

        if sample.perturbed_question:
            question = sample.perturbed_question

        # restore the fields to their original values
        if sample.loaded_fields:
            question_field = sample.loaded_fields["question"]
            context_field = sample.loaded_fields["context"]
            options_field = sample.loaded_fields["options"]
            target_field = sample.loaded_fields["target_column"]

            row_dict = {
                question_field: question,
            }
            if context_field and len(context) > 1:
                row_dict[context_field] = context
            if options_field and len(options) > 1:
                row_dict[options_field] = options

            if target_field and sample.expected_results:
                row_dict[target_field] = (
                    sample.expected_results[0]
                    if isinstance(sample.expected_results, list)
                    else sample.expected_results
                )

        else:
            row_dict = {
                "question": question,
            }

            if context and len(context) > 1:
                row_dict["passage"] = context

            if options and len(options) > 1:
                row_dict["options"] = options

            if sample.expected_results:
                row_dict["answer"] = (
                    sample.expected_results[0]
                    if isinstance(sample.expected_results, list)
                    else sample.expected_results
                )

        return row_dict
