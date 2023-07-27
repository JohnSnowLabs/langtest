import pytest
import pandas as pd

from langtest.datahandler.datasource import (
    CSVDataset,
    ConllDataset,
    HuggingFaceDataset,
    JSONLDataset,
)
from langtest.utils.custom_types.output import (
    NEROutput,
    SequenceClassificationOutput,
    NERPrediction,
    Span,
)
from langtest.utils.custom_types.sample import (
    NERSample,
    QASample,
    SequenceClassificationSample,
    SummarizationSample,
    ToxicitySample,
    TranslationSample,
)


class TestNERDataset:
    """Test cases for ner datasets"""

    sample = NERSample(
        original="I do love KFC",
        test_type="add_context",
        expected_results=NEROutput(
            predictions=[
                NERPrediction(entity="PROD", span=Span(start=10, end=13, word="KFC"))
            ]
        ),
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

    @pytest.mark.parametrize(
        "dataset,params",
        [
            (
                HuggingFaceDataset(dataset_name="wikiann", task="ner"),
                {"subset": "fo", "feature_column": "tokens", "target_column": "ner_tags"},
            ),
            (CSVDataset(file_path="tests/fixtures/tner.csv", task="ner"), {}),
            (ConllDataset(file_path="tests/fixtures/test.conll", task="ner"), {}),
        ],
    )
    def test_load_data(self, dataset, params):
        """"""
        samples = dataset.load_data(**params)

        assert isinstance(samples, list)

        for sample in samples:
            assert isinstance(sample, NERSample)
            assert isinstance(sample.expected_results, NEROutput)

    def test_export_data_csv(self):
        """"""
        dataset = CSVDataset(file_path="tests/fixtures/tner.csv", task="ner")
        dataset.export_data(
            data=[self.sample, self.sample], output_path="/tmp/exported_sample.csv"
        )

        df = pd.read_csv("/tmp/exported_sample.csv")
        saved_sample = df.text[0]

        assert isinstance(saved_sample, str)
        assert " ".join(eval(saved_sample)) == self.sample.original

    def test_export_data_conll(self):
        """"""
        dataset = ConllDataset(file_path="tests/fixtures/test.conll", task="ner")
        dataset.export_data(
            data=[self.sample, self.sample], output_path="/tmp/exported_sample.conll"
        )

        all_tokens, all_labels = [], []
        tokens, labels = [], []
        with open("/tmp/exported_sample.conll", "r") as reader:
            content = reader.read()

            for line in content.strip().split("\n"):
                row = line.strip().split()
                if len(row) == 0:
                    if len(tokens) > 0:
                        all_tokens.append(tokens)
                        all_labels.append(labels)
                        tokens = []
                        labels = []
                    continue
                tokens.append(row[0])
                labels.append(row[-1])

            if len(tokens) != 0:
                all_tokens.append(tokens)
                all_labels.append(labels)

        assert len(all_tokens) == len(all_labels) == 2
        assert " ".join(all_tokens[0]) == self.sample.original

        # assert isinstance(saved_sample, str)
        # assert " ".join(eval(saved_sample)) == self.sample.original


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
                file_path="tests/fixtures/XSum-test-tiny.jsonl",
                task="summarization",
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
            file_path="tests/fixtures/translation-test-tiny.jsonl",
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
            file_path="tests/fixtures/toxicity-test-tiny.jsonl",
            task="toxicity",
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
            file_path="tests/fixtures/TruthfulQA-test-tiny.jsonl",
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
