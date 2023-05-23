import csv
import os
import re
import jsonlines
from abc import ABC, abstractmethod
from typing import Dict, List

from .format import Formatter
from ..utils.custom_types import NEROutput, NERPrediction, NERSample, Sample, SequenceClassificationOutput, \
    SequenceClassificationSample, SequenceLabel, QASample, SummarizationSample


class _IDataset(ABC):
    """Abstract base class for Dataset.

    Defines the load_data method that all subclasses must implement.
    """

    @abstractmethod
    def load_data(self):
        """Load data from the file_path."""
        return NotImplemented

    @abstractmethod
    def export_data(self, data: List[Sample], output_path: str):
        """
        Exports the data to the corresponding format and saves it to 'output_path'.

        Args:
            data (List[Sample]):
                data to export
            output_path (str):
                path to save the data to
        """
        return NotImplementedError()


class DataFactory:
    """Data factory for creating Dataset objects.

    The DataFactory class is responsible for creating instances of the
    correct Dataset type based on the file extension.
    """

    def __init__(
            self,
            file_path: str,
            task: str,
    ) -> None:
        """Initializes DataFactory object.
        Args:
            file_path (str): Path to the dataset.
            task (str): Task to be evaluated.
        """

        self._file_path = file_path
        self._class_map = {cls.__name__.replace(
            'Dataset', '').lower(): cls for cls in _IDataset.__subclasses__()}
        _, self.file_ext = os.path.splitext(self._file_path)
        if len(self.file_ext) > 0:
            self.file_ext = self.file_ext.replace('.', '')
        else:
            self._file_path = self._load_dataset(self._file_path)
            _, self.file_ext = os.path.splitext(self._file_path)
        self.task = task
        self.init_cls = None

    def load(self) -> List[Sample]:
        """Loads the data for the correct Dataset type.

        Returns:
            list[Sample]: Loaded text data.
        """
        self.init_cls = self._class_map[self.file_ext.replace(
            '.', '')](self._file_path, task=self.task)
        return self.init_cls.load_data()

    def export(self, data: List[Sample], output_path: str):
        """
        Exports the data to the corresponding format and saves it to 'output_path'.
        Args:
            data (List[Sample]):
                data to export
            output_path (str):
                path to save the data to
        """
        self.init_cls.export_data(data, output_path)
        
    @classmethod   
    def load_curated_bias(cls, tests_to_filter, file_path)-> List[Sample]:
        data = []
        path = os.path.abspath(__file__)
        if file_path=='BoolQ':
            bias_jsonl = os.path.dirname(path)[: -7]+"/BoolQ/bias.jsonl"
            with jsonlines.open(bias_jsonl) as reader:
                for item in reader:
                    if item['test_type'] in tests_to_filter:
                        data.append(
                            QASample(original_question=item['original_question'], original_context=item.get(
                                'original_context', "-"),perturbed_question=item['perturbed_question'], perturbed_context=item.get(
                                'perturbed_context', "-"), task="question-answering", test_type = item['test_type'], category=item['category'], dataset_name="BoolQ")
                        )
        elif file_path=='XSum':
            bias_jsonl = os.path.dirname(path)[: -7]+"/Xsum/bias.jsonl"
            with jsonlines.open(bias_jsonl) as reader:
                for item in reader:
                    if item['test_type'] in tests_to_filter:           
                        data.append(
                            SummarizationSample(original=item['original'], test_case=item['test_case'], task="summarization", test_type = item['test_type'], category=item['category'], dataset_name="XSum"))
            

        return data

    @classmethod
    def _load_dataset(cls, dataset_name: str):
        """
        Args:
            dataset_name (str): name of the dataset

        Returns:
            str: path to our data
        """
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        datasets_info = {
            'BoolQ-dev-tiny': script_dir[:-7]+'/BoolQ/dev-tiny.jsonl',
            'BoolQ-dev': script_dir[:-7]+'/BoolQ/dev.jsonl',
            'BoolQ-test-tiny': script_dir[:-7]+'/BoolQ/test-tiny.jsonl',
            'BoolQ-test': script_dir[:-7]+'/BoolQ/test.jsonl',
            'BoolQ': script_dir[:-7]+'/BoolQ/combined.jsonl',
            'NQ-open-test': script_dir[:-7]+'/NQ-open/test.jsonl',
            'NQ-open': script_dir[:-7]+'/NQ-open/combined.jsonl',
            'NQ-open-test-tiny': script_dir[:-7]+'/NQ-open/test-tiny.jsonl',
            'XSum-test-tiny' : script_dir[:-7]+'/Xsum/XSum-test-tiny.jsonl'
        }
        return datasets_info[dataset_name]


class ConllDataset(_IDataset):
    """
    Class to handle Conll files. Subclass of _IDataset.
    """

    def __init__(self, file_path: str, task: str) -> None:
        """Initializes ConllDataset object.
        Args:
            file_path (str): Path to the data file.
        """
        super().__init__()
        self._file_path = file_path

        if task != 'ner':
            raise ValueError(
                f'Given task ({task}) is not matched with ner. CoNLL dataset can ne only loaded for ner!')
        self.task = task

    def load_data(self) -> List[NERSample]:
        """Loads data from a CoNLL file.
        Returns:
            List[NERSample]: List of formatted sentences from the dataset.
        """
        data = []
        with open(self._file_path) as f:
            content = f.read()
            docs_strings = re.findall(r"-DOCSTART- \S+ \S+ O", content.strip())
            docs = [i.strip() for i in re.split(
                r"-DOCSTART- \S+ \S+ O", content.strip()) if i != '']
            for d_id, doc in enumerate(docs):
                #  file content to sentence split
                sentences = doc.strip().split('\n\n')

                if sentences == ['']:
                    data.append(([''], ['']))
                    continue

                for sent in sentences:
                    # sentence string to token level split
                    tokens = sent.strip().split('\n')

                    # get annotations from token level split
                    token_list = [t.split() for t in tokens]

                    #  get token and labels from the split
                    ner_labels = []
                    cursor = 0
                    for split in token_list:
                        ner_labels.append(
                            NERPrediction.from_span(
                                entity=split[-1],
                                word=split[0],
                                start=cursor,
                                end=cursor + len(split[0]),
                                doc_id=d_id,
                                doc_name=(docs_strings[d_id] if len(
                                    docs_strings) > 0 else ''),
                                pos_tag=split[1],
                                chunk_tag=split[2]
                            )
                        )
                        # +1 to account for the white space
                        cursor += len(split[0]) + 1

                    original = " ".join(
                        [label.span.word for label in ner_labels])

                    data.append(
                        NERSample(original=original, expected_results=NEROutput(
                            predictions=ner_labels))
                    )

        return data

    def export_data(self, data: List[Sample], output_path: str):
        """
        Exports the data to the corresponding format and saves it to 'output_path'.
        Args:
            data (List[Sample]):
                data to export
            output_path (str):
                path to save the data to
        """
        temp_id = None
        otext = ""
        for i in data:
            text, temp_id = Formatter.process(
                i, output_format='conll', temp_id=temp_id)
            otext += text

        with open(output_path, "wb") as fwriter:
            fwriter.write(bytes(otext, encoding="utf-8"))


class JSONDataset(_IDataset):
    """
    Class to handle JSON dataset files. Subclass of _IDataset.
    """

    def __init__(self, file_path: str):
        """Initializes JSONDataset object.
        Args:
            file_path (str): Path to the data file.
        """
        super().__init__()
        self._file_path = file_path

    def load_data(self) -> List[Sample]:
        """"""
        raise NotImplementedError()

    def export_data(self, data: List[Sample], output_path: str):
        """
        Exports the data to the corresponding format and saves it to 'output_path'.
        Args:
            data (List[Sample]):
                data to export
            output_path (str):
                path to save the data to
        """
        raise NotImplementedError()


class CSVDataset(_IDataset):
    """
    Class to handle CSV files dataset. Subclass of _IDataset.
    """
    COLUMN_NAMES = {
        'text-classification': {
            'text': ['text', 'sentences', 'sentence', 'sample'],
            'label': ['label', 'labels ', 'class', 'classes']
        },
        'ner': {
            'text': ['text', 'sentences', 'sentence', 'sample'],
            'ner': ['label', 'labels ', 'class', 'classes', 'ner_tag', 'ner_tags', 'ner', 'entity'],
            'pos': ['pos_tags', 'pos_tag', 'pos', 'part_of_speech'],
            'chunk': ['chunk_tags', 'chunk_tag']
        }
    }

    def __init__(self, file_path: str, task: str) -> None:
        """Initializes CSVDataset object.
        Args:
            file_path (str):
                Path to the data file.
            task (str):
                Task to be evaluated.
        """
        super().__init__()
        self._file_path = file_path
        self.task = task
        self.delimiter = self._find_delimiter(file_path)
        self.COLUMN_NAMES = self.COLUMN_NAMES[self.task]
        self.column_map = None

    def load_data(self) -> List[Sample]:
        """Loads data from a csv file.
        Returns:
            List[Sample]: List of formatted sentences from the dataset.
        """
        with open(self._file_path, newline='', encoding="utf-8") as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=self.delimiter)

            samples = []
            for sent_indx, row in enumerate(csv_reader):
                if not self.column_map:
                    self.column_map = self._match_column_names(
                        list(row.keys()))

                if self.task == 'ner':
                    samples.append(
                        self._row_to_ner_sample(row, sent_indx)
                    )

                elif self.task == 'text-classification':
                    samples.append(
                        self._row_to_seq_classification_sample(row)
                    )

        return samples

    def export_data(self, data: List[Sample], output_path: str):
        """
        Exports the data to the corresponding format and saves it to 'output_path'.
        Args:
            data (List[Sample]):
                data to export
            output_path (str):
                path to save the data to
        """
        temp_id = None
        otext = ""
        for i in data:
            if isinstance(i, NEROutput):
                text, temp_id = Formatter.process(
                    i, output_format='csv', temp_id=temp_id)
            else:
                text = Formatter.process(i, output_format='csv')
            otext += text

        with open(output_path, "wb") as fwriter:
            fwriter.write(bytes(otext, encoding="utf-8"))

    @staticmethod
    def _find_delimiter(file_path: str) -> property:
        """
        Helper function in charge of finding the delimiter character in a csv file.
        Args:
            file_path (str):
                location of the csv file to load
        Returns:
            property:
        """
        sniffer = csv.Sniffer()
        with open(file_path, encoding="utf-8") as fp:
            first_line = fp.readline()
            delimiter = sniffer.sniff(first_line).delimiter
        return delimiter

    def _row_to_ner_sample(self, row: Dict[str, List[str]], sent_index: int) -> Sample:
        """
        Convert a row from the dataset into a Sample for the NER task.
        Args:
            row (Dict[str, List[str]]):
                single row of the dataset
            sent_index (int):
        Returns:
            Sample:
                row formatted into a Sample object
        """
        assert all(isinstance(value, list) for value in row.values()), \
            ValueError(f"Column ({sent_index}th) values should be list that contains tokens or labels. "
                       "Given CSV file has invalid values")

        token_num = len(row['text'])
        assert all(len(value) == token_num for value in row.values()), \
            ValueError(f"Column ({sent_index}th) values should have same length with number of token in text, "
                       f"which is {token_num}")

        original = " ".join(self.column_map['text'])
        ner_labels = list()
        cursor = 0
        for token_indx in range(len(self.column_map['text'])):
            token = row[self.column_map['text']][token_indx]
            ner_labels.append(
                NERPrediction.from_span(
                    entity=row[self.column_map['ner']][token_indx],
                    word=token,
                    start=cursor,
                    end=cursor + len(token),
                    pos_tag=row[self.column_map['pos']][token_indx]
                    if row.get(self.column_map['pos'], None) else None,
                    chunk_tag=row[self.column_map['chunk']][token_indx]
                    if row.get(self.column_map['chunk'], None) else None,
                )
            )
            cursor += len(token) + 1  # +1 to account for the white space

        return NERSample(
            original=original,
            expected_results=NEROutput(predictions=ner_labels)
        )

    def _row_to_seq_classification_sample(self, row: Dict[str, str]) -> Sample:
        """
        Convert a row from the dataset into a Sample for the text-classification task
        Args:
            row (Dict[str, str]):
                single row of the dataset
        Returns:
            Sample:
                row formatted into a Sample object
        """
        original = row[self.column_map['text']]
        #   label score should be 1 since it is ground truth, required for __eq__
        label = SequenceLabel(label=row[self.column_map['label']], score=1)

        return SequenceClassificationSample(
            original=original,
            expected_results=SequenceClassificationOutput(predictions=[label])
        )

    def _match_column_names(self, column_names: List[str]) -> Dict[str, str]:
        """
        Helper function to map original column into standardized ones.
        Args:
            column_names (List[str]):
                list of column names of the csv file
        Returns:
            Dict[str, str]:
                mapping from the original column names into 'standardized' names
        """
        column_map = {k: None for k in self.COLUMN_NAMES}
        for c in column_names:
            for key, reference_columns in self.COLUMN_NAMES.items():
                if c.lower() in reference_columns:
                    column_map[key] = c

        not_referenced_columns = {
            k: self.COLUMN_NAMES[k] for k, v in column_map.items() if v is None}
        if not_referenced_columns:
            raise OSError(
                f"CSV file is invalid. CSV handler works with template column names!\n"
                f"{', '.join(not_referenced_columns.keys())} column could not be found in header.\n"
                f"You can use following namespaces:\n{not_referenced_columns}"
            )
        return column_map


class JSONLDataset(_IDataset):
    """
    Class to handle BoolQ dataset. Subclass of _IDataset.
    """

    def __init__(self, file_path: str, task: str) -> None:
        """Initializes BOOLQDataset object.
        Args:
            file_path (str): Path to the data file.
        """
        super().__init__()
        self._file_path = file_path
        self.task = task

    def load_data(self):
        """Loads data from a JSONL file.
        Returns:
            list[QASample]: Loaded text data.
        """

        data = []
        with jsonlines.open(self._file_path) as reader:
            for item in reader:
                if (self.task=='question-answering'):
                    expected_results = item.get("answer_and_def_correct_predictions", item.get("answer", None))
                    if isinstance(expected_results, str) or isinstance(expected_results, bool): expected_results = [str(expected_results)]

                    data.append(
                        QASample(
                            original_question = item['question'],
                            original_context= item.get('passage', "-"),
                            expected_results = expected_results,
                            task=self.task,
                            dataset_name=self._file_path.split('/')[-2]
                            )
                    )

                elif (self.task=='summarization'):
                    expected_results = item.get("summary",None)
                    if isinstance(expected_results, str) or isinstance(expected_results, bool): expected_results = [str(expected_results)]
                    data.append(
                    SummarizationSample(
                        original = item['document'],
                        expected_results=expected_results,
                        task=self.task,
                        dataset_name=self._file_path.split('/')[-2]
                        )
                )
                
        return data

    def export_data(self, data: List[Sample], output_path: str):
        """
        Exports the data to the corresponding format and saves it to 'output_path'.
        Args:
            data (List[Sample]):
                data to export
            output_path (str):
                path to save the data to
        """
        raise NotImplementedError()
