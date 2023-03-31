import csv
import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict

from .format import Formatter
from ..utils.custom_types import NEROutput, SequenceClassificationOutput, NERPrediction, SequenceLabel, Sample


class _IDataset(ABC):
    """Abstract base class for Dataset.

    Defines the load_data method that all subclasses must implement.
    """

    @abstractmethod
    def load_data(self):
        """Load data from the file_path."""
        return NotImplemented

    @abstractmethod
    def export_data(self):
        return NotImplemented


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
        """

        self._file_path = file_path
        self._class_map = {cls.__name__.replace('Dataset', '').lower(): cls for cls in _IDataset.__subclasses__()}
        _, self.file_ext = os.path.splitext(self._file_path)
        self.task = task

    def load(self):
        """Loads the data for the correct Dataset type.

        Returns:
            list[str]: Loaded text data.
        """
        self.init_cls = self._class_map[self.file_ext.replace('.', '')](self._file_path, task=self.task)
        return self.init_cls.load_data()

    def export(self, data: List[Sample], output_path: str):
        return self.init_cls.export_data(data, output_path)


class ConllDataset(_IDataset):
    """Class to handle Conll files. Subclass of _IDataset.
    """

    def __init__(self, file_path: str, task: str) -> None:
        """Initializes ConllDataset object.

        Args:
            file_path (str): Path to the data file.
        """
        super().__init__()
        self._file_path = file_path

        if task != 'ner':
            raise OSError(f'Given task ({task}) is not matched with ner. CoNLL dataset can ne only loaded for ner!')
        self.task = task

    def load_data(self) -> List[Sample]:
        """Loads data from a CoNLL file.

        Returns:
            list: List of sentences in the dataset.
        """
        data = []
        with open(self._file_path) as f:
            content = f.read()
            docs_strings = re.findall(r"-DOCSTART- \S+ \S+ O", content.strip())
            docs = [i.strip() for i in re.split(r"-DOCSTART- \S+ \S+ O", content.strip()) if i != '']
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
                                doc_name=(docs_strings[d_id] if len(docs_strings) > 0 else '') ,
                                pos_tag=split[1],
                                chunk_tag=split[2]
                            )
                        )
                        cursor += len(split[0]) + 1  # +1 to account for the white space

                    original = " ".join([label.span.word for label in ner_labels])

                    data.append(
                        Sample(original=original, expected_results=NEROutput(predictions=ner_labels))
                    )

        return data

    def export_data(self, data: List[Sample], output_path: str):
        temp_id = None
        otext = ""
        for i in data:
            text, temp_id = Formatter.process(i, format='conll', temp_id=temp_id)
            otext += text

        with open(output_path, "wb") as fwriter:
            fwriter.write(bytes(otext, encoding="utf-8"))




class JSONDataset(_IDataset):
    """Class to handle JSON dataset files. Subclass of _IDataset.
    """

    def __init__(self, file_path) -> None:
        """Initializes JSONDataset object.

        Args:
            file_path (str): Path to the data file.
        """
        super().__init__()
        self._file_path = file_path

    def load_data(self):
        pass

    def export_data(self):
        pass


class CSVDataset(_IDataset):
    """Class to handle CSV files dataset. Subclass of _IDataset.
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
            file_path (str): Path to the data file.
        """
        super().__init__()
        self._file_path = file_path
        self.task = task
        self.delimiter = self.find_delimiter(file_path)
        self.COLUMN_NAMES = self.COLUMN_NAMES[self.task]
        self.column_map = None

    def load_data(self) -> List[Sample]:
        """Loads data from a csv file.

        Returns:

        """
        with open(self._file_path, newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=self.delimiter)

            samples = []
            for sent_indx, row in enumerate(csv_reader):
                if not self.column_map:
                    self.column_map = self.match_column_names(list(row.keys()))

                if self.task == 'ner':
                    samples.append(
                        self.row_to_ner_sample(row, sent_indx)
                    )

                elif self.task == 'text-classification':
                    samples.append(
                        self.row_to_seq_classification_sample(row)
                    )

        return samples

    def export_data(self, data: List[Sample], output_path: str):
        temp_id = None
        otext = ""
        for i in data:
            if isinstance(i, NEROutput):
                text, temp_id = Formatter.process(i, format='csv', temp_id=temp_id)
            else:
                text = Formatter.process(i, format='csv')
            otext += text

        with open(output_path, "wb") as fwriter:
            fwriter.write(bytes(otext, encoding="utf-8"))


    #   helpers
    @staticmethod
    def find_delimiter(file_path):
        sniffer = csv.Sniffer()
        with open(file_path) as fp:
            first_line = fp.readline()
            delimiter = sniffer.sniff(first_line).delimiter
        return delimiter

    def row_to_ner_sample(self, row: Dict[str, List[str]], sent_indx: int) -> Sample:
        assert all(isinstance(value, list) for value in row.values()), \
            ValueError(f"Column ({sent_indx}th) values should be list that contains tokens or labels. "
                       "Given CSV file has invalid values")

        token_num = len(row['text'])
        assert all(len(value) == token_num for value in row.values()), \
            ValueError(f"Column ({sent_indx}th) values should have same length with number of token in text, "
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

        return Sample(original=original, expected_results=NEROutput(predictions=ner_labels))

    def row_to_seq_classification_sample(self, row: Dict[str, str]) -> Sample:

        original = row[self.column_map['text']]
        #   label score should be 1 since it is ground truth, required for __eq__
        label = SequenceLabel(label=row[self.column_map['label']], score=1)

        return Sample(original=original, expected_results=SequenceClassificationOutput(predictions=[label]))

    def match_column_names(self, column_names: List[str]):
        column_map = {k: None for k in self.COLUMN_NAMES}
        for c in column_names:
            for key, reference_columns in self.COLUMN_NAMES.items():
                if c.lower() in reference_columns:
                    column_map[key] = c

        not_referenced_columns = {k: self.COLUMN_NAMES[k] for k, v in column_map.items() if v is None}
        if not_referenced_columns:
            raise OSError(
                f"CSV file is invalid. CSV handler works with template column names!\n"
                f"{', '.join(not_referenced_columns.keys())} column could not be found in header.\n"
                f"You can use following namespaces:\n{not_referenced_columns}"
            )
        return column_map
