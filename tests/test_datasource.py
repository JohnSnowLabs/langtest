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
    "dataset,feature_col,target_col",
    [
        (
            CSVDataset(file_path="tests/fixtures/tner.csv", task="ner"),
            "tokens",
            "ner_tags",
        ),
        (
            ConllDataset(file_path="tests/fixtures/test.conll", task="ner"),
            "text",
            "labels",
        ),
    ],
)
class TestNERDataset:
    """Test cases for ner datasets"""

    def test_load_raw_data(self, dataset, feature_col, target_col):
        """"""
        raw_data = dataset.load_raw_data()

        assert isinstance(raw_data, list)

        for sample in raw_data:
            assert isinstance(sample[feature_col], list)
            assert isinstance(sample[target_col], list)
            assert len(sample[feature_col]) == len(sample[target_col])

            for token in sample[feature_col]:
                assert isinstance(token, str)

            for label in sample[target_col]:
                assert isinstance(label, str)

    def test_load_data(self, dataset, feature_col, target_col):
        """"""
        samples = dataset.load_data()

        assert isinstance(samples, list)

        for sample in samples:
            assert isinstance(sample, NERSample)
            assert isinstance(sample.expected_results, NEROutput)


@pytest.mark.parametrize(
    "dataset,feature_col,target_col",
    [
        (
            CSVDataset(
                file_path="tests/fixtures/text_classification.csv",
                task="text-classification",
            ),
            "text",
            "label",
        ),
        (
            HuggingFaceDataset(dataset_name="dbrd", task="text-classification"),
            "text",
            "label",
        ),
    ],
)
class TestTextClassificationDataset:
    """Test cases for text classification datasets"""

    def test_load_raw_data(self, dataset, feature_col, target_col):
        """"""
        raw_data = dataset.load_raw_data()

        for sample in raw_data:
            assert isinstance(sample[feature_col], str)
            assert isinstance(sample[target_col], int)

    def test_load_data(self, dataset, feature_col, target_col):
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
    "dataset,feature_col,target_col",
    [
        (
            HuggingFaceDataset(
                dataset_name="JulesBelveze/tldr_news", task="summarization"
            ),
            "content",
            "headline",
        ),
        (
            JSONLDataset(
                file_path="langtest/data/Xsum/Xsum-test-tiny.jsonl", task="summarization"
            ),
            "document",
            "summary",
        ),
        (
            JSONLDataset(file_path="/tmp/summarization_1.jsonl", task="summarization"),
            "text",
            "summary",
        ),
    ],
)
class TestSummarizationDataset:
    """Test cases for summarization datasets"""

    def test_load_raw_data(self, dataset, feature_col, target_col):
        """"""
        if isinstance(dataset, HuggingFaceDataset):
            raw_data = dataset.load_raw_data(
                feature_column=feature_col, target_column=target_col, split="test[:30]"
            )
        else:
            raw_data = dataset.load_raw_data()

        for sample in raw_data:
            assert isinstance(sample[feature_col], str)
            assert isinstance(sample[target_col], str)

    def test_load_data(self, dataset, feature_col, target_col):
        """"""
        if isinstance(dataset, HuggingFaceDataset):
            samples = dataset.load_data(
                feature_column=feature_col, target_column=target_col, split="test[:30]"
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
