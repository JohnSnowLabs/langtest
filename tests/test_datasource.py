import pytest

from langtest.datahandler.datasource import (
    CSVDataset,
    ConllDataset,
    HuggingFaceDataset,
    JSONLDataset,
)
from langtest.utils.custom_types.output import NEROutput, SequenceClassificationOutput
from langtest.utils.custom_types.sample import (
    NERSample,
    SequenceClassificationSample,
    SummarizationSample,
    ToxicitySample,
    QASample,
    TranslationSample,
)


@pytest.mark.parametrize(
    "dataset",
    [
        CSVDataset(file_path="tests/fixtures/tner.csv", task="ner"),
        ConllDataset(file_path="tests/fixtures/test.conll", task="ner"),
    ],
)
class TestNERDataset:
    """Test cases for ner datasets"""

    def test_load_raw_data(self, dataset):
        """"""
        raw_data = dataset.load_raw_data()

        assert isinstance(raw_data, list)

        for sample in raw_data:
            assert isinstance(sample["text"], list)
            assert isinstance(sample["labels"], list)
            assert len(sample["text"]) == len(sample["labels"])

            for token in sample["text"]:
                assert isinstance(token, str)

            for label in sample["labels"]:
                assert isinstance(label, str)

    def test_load_data(self, dataset):
        """"""
        samples = dataset.load_data()

        assert isinstance(samples, list)

        for sample in samples:
            assert isinstance(sample, NERSample)
            assert isinstance(sample.expected_results, NEROutput)


@pytest.mark.parametrize(
    "dataset",
    [
        CSVDataset(
            file_path="tests/fixtures/text_classification.csv", task="text-classification"
        ),
        HuggingFaceDataset(dataset_name="dbrd", task="text-classification"),
    ],
)
class TestTextClassificationDataset:
    """Test cases for text classification datasets"""

    def test_load_raw_data(self, dataset):
        """"""
        raw_data = dataset.load_raw_data()

        for sample in raw_data:
            assert isinstance(sample["text"], str)
            assert isinstance(sample["labels"], int)

    def test_load_data(self, dataset):
        """"""
        if isinstance(dataset, HuggingFaceDataset):
            samples = dataset.load_data(split="test[:30]")
        else:
            samples = dataset.load_data()

        assert isinstance(samples, list)

        for sample in samples:
            assert isinstance(sample, SequenceClassificationSample)
            assert isinstance(sample.expected_results, SequenceClassificationOutput)


@pytest.mark.parametrize(
    "dataset",
    [
        HuggingFaceDataset(dataset_name="JulesBelveze/tldr_news", task="summarization"),
        JSONLDataset(
            file_path="langtest/data/Xsum/Xsum-test-tiny.jsonl", task="summarization"
        ),
    ],
)
class TestSummarizationDataset:
    """Test cases for summarization datasets"""

    def test_load_raw_data(self, dataset):
        """"""
        if isinstance(dataset, HuggingFaceDataset):
            raw_data = dataset.load_raw_data(
                feature_column="content", target_column="headline", split="test[:30]"
            )
        else:
            raw_data = dataset.load_raw_data()

        for sample in raw_data:
            assert isinstance(sample["document"], str)
            assert isinstance(sample["summary"], str)

    def test_load_data(self, dataset):
        """"""
        if isinstance(dataset, HuggingFaceDataset):
            samples = dataset.load_data(
                feature_column="content", target_column="headline", split="test[:30]"
            )
        else:
            samples = dataset.load_data()

        assert isinstance(samples, list)

        for sample in samples:
            assert isinstance(sample, SummarizationSample)


@pytest.mark.parametrize(
    "dataset",
    [
        JSONLDataset(
            file_path="langtest/data/Translation/translation-test-tiny.jsonl",
            task="translation",
        )
    ],
)
class TestTranslationDataset:
    """Test cases for translation datasets"""

    def test_load_raw_data(self, dataset):
        """"""
        raw_data = dataset.load_raw_data()
        assert isinstance(raw_data, list)

        for sample in raw_data:
            assert isinstance(sample["sourceString"], str)

    def test_load_data(self, dataset):
        """"""
        samples = dataset.load_data()

        assert isinstance(samples, list)

        for sample in samples:
            assert isinstance(sample, TranslationSample)


@pytest.mark.parametrize(
    "dataset",
    [
        JSONLDataset(
            file_path="langtest/data/toxicity/toxicity-test-tiny.jsonl", task="toxicity"
        )
    ],
)
class TestToxicityDataset:
    """Test cases for toxicity datasets"""

    def test_load_raw_data(self, dataset):
        """"""
        raw_data = dataset.load_raw_data()
        assert isinstance(raw_data, list)

        for sample in raw_data:
            assert isinstance(sample["text"], str)

    def test_load_data(self, dataset):
        """"""
        samples = dataset.load_data()

        assert isinstance(samples, list)

        for sample in samples:
            assert isinstance(sample, ToxicitySample)


@pytest.mark.parametrize(
    "dataset",
    [
        JSONLDataset(
            file_path="langtest/data/TruthfulQA/TruthfulQA-test-tiny.jsonl",
            task="question-answering",
        )
    ],
)
class TestQADataset:
    """Test cases for QA datasets"""

    def test_load_raw_data(self, dataset):
        """"""
        raw_data = dataset.load_raw_data()
        assert isinstance(raw_data, list)

        for sample in raw_data:
            assert isinstance(sample["question"], str)
            assert isinstance(sample["answer"], list)

    def test_load_data(self, dataset):
        """"""
        samples = dataset.load_data()

        assert isinstance(samples, list)

        for sample in samples:
            assert isinstance(sample, QASample)
