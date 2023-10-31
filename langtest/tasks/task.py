import re
from abc import ABC, abstractmethod
from typing import Union
from langtest.modelhandler import ModelAPI, LANGCHAIN_HUBS
from langtest.errors import Errors, ColumnNameError

from langtest.utils import custom_types as samples
from langtest.utils.custom_types.predictions import NERPrediction


class BaseTask(ABC):
    """Abstract base class for all tasks."""

    task_registry = {}
    _name = None
    sample_class = None

    @classmethod
    @abstractmethod
    def create_sample(cls, *args, **kwargs) -> samples.Sample:
        """Run the task."""
        pass

    @classmethod
    def load_model(cls, model_path: str, model_hub: str, *args, **kwargs):
        """Load the model."""

        models = ModelAPI.model_registry

        base_hubs = list(models.keys())
        base_hubs.remove("llm")
        supported_hubs = base_hubs + list(LANGCHAIN_HUBS.keys())

        if model_hub not in supported_hubs:
            raise AssertionError(Errors.E042.format(supported_hubs=supported_hubs))

        if model_hub in LANGCHAIN_HUBS:
            # LLM models
            cls.model = models["llm"][cls._name].load_model(
                hub=model_hub, path=model_path, *args, **kwargs
            )
        else:
            # JSL, Huggingface, and Spacy models
            cls.model = models[model_hub][cls._name].load_model(
                path=model_path, *args, **kwargs
            )
        return cls.model

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        task_name = re.sub(
            r"(?<=[a-z])(?=[A-Z][a-z])", "-", cls.__name__.replace("Task", "")
        ).lower()

        cls.task_registry[task_name] = cls

    def __eq__(self, __value: object) -> bool:
        """Check if the task is equal to another task."""
        if isinstance(__value, str):
            return self.__class__.__name__.replace("Task", "").lower() == __value.lower()
        return super().__eq__(__value)

    def column_mapping(self, item_keys, columns_names, *args, **kwargs):
        """Return the column mapping."""

        column_mapper = {}
        for item in item_keys:
            if columns_names and item in columns_names:
                column_mapper[item] = item
                continue
            for key in self._default_col:
                if item.lower() in self._default_col[key]:
                    column_mapper[key] = item
                    break
        # column_mapper  values set should be in item_keys
        if not all(
            [
                value.lower() in map(lambda x: x.lower(), item_keys)
                and key in columns_names
                for key, value in column_mapper.items()
            ]
        ):
            raise ColumnNameError(list(self._default_col), item_keys)
        return column_mapper

    @property
    def get_sample_cls(self):
        """Return the sample class."""
        if self.sample_class:
            return self.sample_class
        return None


class TaskManager:
    """Task manager."""

    def __init__(self, task_name: str):
        if task_name not in BaseTask.task_registry:
            raise AssertionError(
                Errors.E043.format(l=list(BaseTask.task_registry.keys()))
            )
        self.__task_name = task_name
        self.__task: BaseTask = BaseTask.task_registry[task_name]()

    def create_sample(self, *args, **kwargs):
        """Add a task to the task manager."""
        # filter out the key with contains column name
        if "feature_column" in kwargs:
            column_names = kwargs["feature_column"]
            if isinstance(column_names, dict):
                kwargs.pop("feature_column")
                kwargs.update(column_names)

        return self.__task.create_sample(*args, **kwargs)

    def model(self, *args, **kwargs) -> "ModelAPI":
        """Add a task to the task manager."""
        return self.__task.load_model(*args, **kwargs)

    def __eq__(self, __value: str) -> bool:
        """Check if the task is equal to another task."""
        return self.__task_name == __value.lower()

    def __hash__(self) -> int:
        """Return the hash of the task name."""
        return hash(self.__task_name)

    def __str__(self) -> str:
        """Return the task name."""
        return self.__task_name

    @property
    def task_name(self):
        """Return the task name."""
        return self.__task_name

    @property
    def get_sample_class(self):
        """
        Return the sample class.

        Returns:
            Sample: Sample class
        """
        return self.__task.get_sample_cls


class NER(BaseTask):
    """Named Entity Recognition task."""

    _name = "ner"
    _default_col = {
        "text": ["text", "sentences", "sentence", "sample", "tokens"],
        "ner": [
            "label",
            "labels ",
            "class",
            "classes",
            "ner_tag",
            "ner_tags",
            "ner",
            "entity",
        ],
        "pos": ["pos_tags", "pos_tag", "pos", "part_of_speech"],
        "chunk": ["chunk_tags", "chunk_tag"],
    }
    sample_class = samples.NERSample

    def create_sample(
        cls,
        row_data: dict,
        feature_column="text",
        target_column: str = "ner",
        pos_tag: str = "pos",
        chunk_tag: str = "chunk_tag",
        *args,
        **kwargs,
    ) -> samples.NERSample:
        """Create a sample."""
        keys = list(row_data.keys())

        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(
            keys, [feature_column, target_column, pos_tag, chunk_tag]
        )

        for key, value in row_data.items():
            if isinstance(value, str):
                row_data[key] = eval(value)
            else:
                row_data[key] = value

        original = " ".join(row_data[column_mapper[feature_column]])
        ner_labels = list()
        cursor = 0
        for token_indx in range(len(row_data[column_mapper[feature_column]])):
            token = row_data[column_mapper[feature_column]][token_indx]
            ner_labels.append(
                NERPrediction.from_span(
                    entity=row_data[column_mapper[target_column]][token_indx],
                    word=token,
                    start=cursor,
                    end=cursor + len(token),
                    pos_tag=row_data[column_mapper[pos_tag]][token_indx]
                    if pos_tag in column_mapper and column_mapper[pos_tag] in row_data
                    else None,
                    chunk_tag=row_data[column_mapper[chunk_tag]][token_indx]
                    if chunk_tag in column_mapper and column_mapper[chunk_tag] in row_data
                    else None,
                )
            )
            cursor += len(token) + 1  # +1 to account for the white space

        expected_results = samples.NEROutput(predictions=ner_labels)

        return samples.NERSample(original=original, expected_results=expected_results)


class TextClassification(BaseTask):
    """Text Classification task."""

    _name = "textclassification"
    _default_col = {
        "text": ["text", "sentences", "sentence", "sample"],
        "label": ["label", "labels ", "class", "classes"],
    }
    sample_class = samples.SequenceClassificationSample

    def create_sample(
        cls,
        row_data: dict,
        feature_column="text",
        target_column: Union[samples.SequenceLabel, str] = "label",
    ) -> samples.SequenceClassificationSample:
        """Create a sample."""
        keys = list(row_data.keys())
        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(keys, [feature_column, target_column])

        labels = row_data.get(column_mapper[target_column])

        if isinstance(labels, samples.SequenceLabel):
            labels = [labels]
        elif isinstance(labels, list):
            labels = [
                samples.SequenceLabel(label=label, score=1.0)
                if isinstance(label, str)
                else label
                for label in labels
            ]
        else:
            labels = [samples.SequenceLabel(label=labels, score=1.0)]

        return samples.SequenceClassificationSample(
            original=row_data[column_mapper[feature_column]],
            expected_results=samples.SequenceClassificationOutput(predictions=labels),
        )


class QuestionAnswering(BaseTask):
    """Question Answering task."""

    _name = "qa"
    _default_col = {
        "text": ["question"],
        "context": ["context", "passage"],
        "answer": ["answer", "answer_and_def_correct_predictions"],
    }
    sample_class = samples.QASample

    def create_sample(
        cls,
        row_data: dict,
        dataset_name: str = "qa",
        question: str = "text",
        context: str = "context",
        target_column: str = "answer",
    ) -> samples.QASample:
        """Create a sample."""
        keys = list(row_data.keys())
        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(keys, [question, context, target_column])

        expected_results = (
            row_data.get(column_mapper[target_column], None)
            if target_column in column_mapper
            else None
        )
        if isinstance(expected_results, str) or isinstance(expected_results, bool):
            expected_results = [str(expected_results)]

        return samples.QASample(
            original_question=row_data[column_mapper[question]],
            original_context=row_data.get(column_mapper.get(context, "-"), "-"),
            expected_results=expected_results,
            dataset_name=dataset_name,
        )


class Summarization(BaseTask):
    """Summarization task."""

    _name = "summarization"
    _default_col = {"text": ["text", "document"], "summary": ["summary"]}
    sample_class = samples.SummarizationSample

    def create_sample(
        cls,
        row_data: dict,
        feature_column="document",
        target_column="summary",
        dataset_name: str = "default_summarization_prompt",
    ) -> samples.SummarizationSample:
        """Create a sample."""
        keys = list(row_data.keys())

        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(keys, [feature_column, target_column])

        expected_results = row_data.get(column_mapper[target_column])
        if isinstance(expected_results, str) or isinstance(expected_results, bool):
            expected_results = [str(expected_results)]

        return samples.SummarizationSample(
            original=row_data[column_mapper[feature_column]],
            expected_results=expected_results,
            dataset_name=dataset_name,
        )


class Translation(BaseTask):
    """Translation task."""

    _name = "translation"
    _default_col = {"text": ["text", "original", "sourcestring"]}
    sample_class = samples.TranslationSample

    def create_sample(
        cls, row_data: dict, feature_column="text", dataset_name: str = "translation"
    ) -> samples.TranslationSample:
        """Create a sample."""
        keys = list(row_data.keys())
        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(keys, [feature_column])

        return samples.TranslationSample(
            original=row_data[column_mapper[feature_column]],
            dataset_name=dataset_name,
        )


class Toxicity(BaseTask):
    """Toxicity task."""

    _name = "toxicity"
    _default_col = {"text": ["text"]}
    sample_class = samples.ToxicitySample

    def create_sample(
        cls, row_data: dict, feature_column: str = "text", dataset_name: str = "toxicity"
    ) -> samples.ToxicitySample:
        """Create a sample."""

        keys = list(row_data.keys())
        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(keys, [feature_column])

        return samples.ToxicitySample(
            prompt=row_data[column_mapper[feature_column]],
            dataset_name=dataset_name,
        )


class Security(BaseTask):
    """Security task."""

    _name = "security"
    _default_col = {"text": ["text", "prompt"]}
    sample_class = samples.SecuritySample

    def create_sample(
        cls, row_data: dict, feature_column="text", dataset_name: str = "security"
    ) -> samples.SecuritySample:
        """Create a sample."""

        keys = list(row_data.keys())

        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(keys, [feature_column])

        return samples.SecuritySample(
            prompt=row_data[column_mapper[feature_column]],
            dataset_name=dataset_name,
        )


class ClinicalTests(BaseTask):
    """Clinical Tests task."""

    _name = "clinicaltests"
    _default_col = {
        "Patient info A": [
            "Patient info A",
            "patient info a",
        ],
        "Patient info B": [
            "Patient info B",
            "patient info b",
        ],
        "Diagnosis": [
            "Diagnosis",
            "diagnosis",
        ],
    }
    sample_class = samples.ClinicalSample

    def create_sample(
        cls,
        row_data: dict,
        patient_info_A: str = "Patient info A",
        patient_info_B: str = "Patient info B",
        diagnosis: str = "Diagnosis",
        dataset_name: str = "clinicaltests",
    ) -> samples.ClinicalSample:
        """Create a sample."""

        keys = list(row_data.keys())
        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(
            keys, [patient_info_A, patient_info_B, diagnosis]
        )

        return samples.ClinicalSample(
            patient_info_A=row_data[column_mapper[patient_info_A]],
            patient_info_B=row_data[column_mapper[patient_info_B]],
            diagnosis=row_data[column_mapper[diagnosis]],
            dataset_name=dataset_name,
        )


class DisinformationTest(BaseTask):
    """Disinformation Test task."""

    _name = "disinformationtest"

    _default_col = {
        "hypothesis": ["hypothesis", "thesis"],
        "statements": ["statements", "headlines"],
    }
    sample_class = samples.DisinformationSample

    def create_sample(
        cls,
        row_data: dict,
        hypothesis: str = "hypothesis",
        statements: str = "statements",
        dataset_name: str = "disinformationtest",
    ) -> samples.DisinformationSample:
        """Create a sample."""

        keys = list(row_data.keys())

        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(keys, [hypothesis, statements])

        return samples.DisinformationSample(
            hypothesis=row_data[column_mapper[hypothesis]],
            statements=row_data[column_mapper[statements]],
            dataset_name=dataset_name,
        )


class Political(BaseTask):
    """Political task."""

    _name = "political"
    _default_col = {"text": ["text", "question"]}
    sample_class = samples.LLMAnswerSample

    def create_sample(
        cls, row_data: dict, feature_column="question", dataset_name: str = "political"
    ) -> samples.LLMAnswerSample:
        """Create a sample."""
        keys = list(row_data.keys())

        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(keys, [feature_column])

        return samples.LLMAnswerSample(
            question=row_data[column_mapper[feature_column]],
            dataset_name=dataset_name,
        )


class WinoBias(BaseTask):
    """WinoBias task."""

    _name = "winobias"
    _default_col = {
        "text": ["text", "prompt"],
        "options": ["options", "choices"],
    }
    sample_class = samples.WinoBiasSample

    def create_sample(
        cls,
        row_data: dict,
        feature_column="text",
        options: str = "options",
        dataset_name: str = "winobias",
    ) -> samples.WinoBiasSample:
        """Create a sample."""
        keys = list(row_data.keys())

        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(keys, [feature_column, options])

        return samples.WinoBiasSample(
            masked_text=row_data[column_mapper[feature_column]],
            options=row_data[column_mapper[options]],
            dataset_name=dataset_name,
        )


class LegalTests(BaseTask):
    """Legal task."""

    _name = "legal"
    _default_col = (
        {
            "case": ["case"],
            "legal-claim": ["legal-claim"],
            "legal_conclusion_a": ["legal_conclusion_a"],
            "legal_conclusion_b": ["legal_conclusion_b"],
            "correct_choice": ["correct_choice"],
        },
    )
    sample_class = samples.LegalSample

    def create_sample(
        cls,
        row_data: dict,
        feature_column="case",
        legal_claim: str = "legal-claim",
        legal_conclusion_A: str = "legal_conclusion_a",
        legal_conclusion_B: str = "legal_conclusion_b",
        correct_choice: str = "correct_choice",
        dataset_name: str = "legal",
    ) -> samples.LegalSample:
        """Create a sample."""
        keys = list(row_data.keys())

        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(
            keys,
            [
                feature_column,
                legal_claim,
                legal_conclusion_A,
                legal_conclusion_B,
                correct_choice,
            ],
        )

        return samples.LegalSample(
            case=row_data[column_mapper[feature_column]],
            legal_claim=row_data[column_mapper[legal_claim]],
            legal_conclusion_A=row_data[column_mapper[legal_conclusion_A]],
            legal_conclusion_B=row_data[column_mapper[legal_conclusion_B]],
            correct_conlusion=row_data[column_mapper[correct_choice]],
            dataset_name=dataset_name,
        )


class FactualityTest(BaseTask):
    """Factuality task."""

    _name = "factuality"
    _default_col = {
        "article_sent": ["article_sent"],
        "correct_sent": ["correct_sent"],
        "incorrect_sent": ["incorrect_sent"],
    }
    sample_class = samples.FactualitySample

    def create_sample(
        cls,
        row_data: dict,
        feature_column="article_sent",
        correct_sent: str = "correct_sent",
        incorrect_sent: str = "incorrect_sent",
        dataset_name: str = "factuality",
    ) -> samples.FactualitySample:
        """Create a sample."""
        keys = list(row_data.keys())

        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(
            keys, [feature_column, correct_sent, incorrect_sent]
        )

        return samples.FactualitySample(
            article_sent=row_data[column_mapper[feature_column]],
            correct_sent=row_data[column_mapper[correct_sent]],
            incorrect_sent=row_data[column_mapper[incorrect_sent]],
            dataset_name=dataset_name,
        )


class SensitivityTest(BaseTask):
    """Sensitivity task."""

    _name = "sensitivity"
    _default_col = {"text": ["text", "question"]}
    sample_class = samples.SensitivitySample

    def create_sample(
        cls, row_data: dict, feature_column="text", *args, **kwargs
    ) -> samples.SensitivitySample:
        """Create a sample."""
        keys = list(row_data.keys())

        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(keys, [feature_column])

        return samples.SensitivitySample(
            original=row_data[column_mapper[feature_column]],
            *args,
            **kwargs,
        )


class CrowsPairs(BaseTask):
    """Crows Pairs task."""

    _name = "crowspairs"
    _default_col = {
        "text": ["text", "sentence"],
        "mask1": ["mask1"],
        "mask2": ["mask2"],
    }
    sample_class = samples.CrowsPairsSample

    def create_sample(
        cls,
        row_data: dict,
        feature_column="sentence",
        mask1: str = "mask1",
        mask2: str = "mask2",
        *args,
        **kwargs,
    ) -> samples.CrowsPairsSample:
        """Create a sample."""
        keys = list(row_data.keys())

        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(keys, [feature_column, mask1, mask2])

        return samples.CrowsPairsSample(
            sentence=row_data[column_mapper[feature_column]],
            mask1=row_data[column_mapper[mask1]],
            mask2=row_data[column_mapper[mask2]],
        )


class Stereoset(BaseTask):
    """StereoSet task."""

    _name = "stereoset"
    _default_col = {
        "text": ["text", "sentence"],
        "mask1": ["mask1"],
        "mask2": ["mask2"],
    }
    sample_class = samples.StereoSetSample

    def create_sample(
        cls,
        row_data: dict,
        bias_type: str = "bias_type",
        test_type: str = "type",
        target_column: str = "target",
        context: str = "context",
        sent_stereo: str = "stereotype",
        sent_antistereo: str = "anti-stereotype",
        sent_unrelated: str = "unrelated",
        *args,
        **kwargs,
    ) -> samples.StereoSetSample:
        """Create a sample."""
        keys = list(row_data.keys())

        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(
            keys,
            [
                bias_type,
                test_type,
                target_column,
                context,
                sent_stereo,
                sent_antistereo,
                sent_unrelated,
            ],
        )

        return samples.StereoSetSample(
            test_type=row_data[column_mapper[test_type]],
            bias_type=row_data[column_mapper[bias_type]],
            target=row_data[column_mapper[target_column]],
            context=row_data[column_mapper[context]],
            sent_stereo=row_data[column_mapper[sent_stereo]],
            sent_antistereo=row_data[column_mapper[sent_antistereo]],
            sent_unrelated=row_data[column_mapper[sent_unrelated]],
        )


class SycophancyTest(BaseTask):
    """Sycophancy task."""

    _name = "sycophancytest"
    _default_col = {
        "text": ["text", "question"],
        "answer": ["answer", "answer_and_def_correct_predictions"],
    }
    sample_class = samples.SycophancySample

    def create_sample(
        cls,
        row_data: dict,
        feature_column="question",
        target_column="answer",
        dataset_name: str = "",
        *args,
        **kwargs,
    ) -> samples.SycophancySample:
        """Create a sample."""
        keys = list(row_data.keys())

        # auto-detect the default column names from the row_data
        column_mapper = cls.column_mapping(keys, [feature_column, target_column])

        return samples.SycophancySample(
            original_question=row_data[column_mapper[feature_column]],
            ground_truth=row_data[column_mapper[target_column]],
            dataset_name=dataset_name,
            *args,
            **kwargs,
        )
