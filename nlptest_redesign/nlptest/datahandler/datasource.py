from abc import ABC, abstractmethod
import os
import pandas as pd

class _IDataset(ABC):
    """Abstract base class for Dataset.

    Defines the load_data method that all subclasses must implement.
    """

    @abstractmethod
    def load_data(self):
        """Load data from the file_path.
        """
        return NotImplemented


class DataFactory:
    """Data factory for creating Dataset objects.

    The DataFactory class is responsible for creating instances of the
    correct Dataset type based on the file extension.
    """

    def __init__(self, file_path) -> None:
        """Initializes DataFactory object.

        Args:
            file_path (str): Path to the dataset.
        """

        self._file_path = file_path
        self._class_map = {cls.__name__.replace('Dataset', '').lower(): cls for cls in _IDataset.__subclasses__()}
        _, self.file_ext = os.path.splitext(self._file_path)

    def load(self):
        """Loads the data for the correct Dataset type.

        Returns:
            list[str]: Loaded text data.
        """
        return self._class_map[self.file_ext.replace('.', '')](self._file_path).load_data()


class ConllDataset(_IDataset):
    """Class to handle Conll files. Subclass of _IDataset.
    """

    def __init__(self, file_path) -> None:
        """Initializes ConllDataset object.

        Args:
            file_path (str): Path to the data file.
        """
        super().__init__()
        self._file_path = file_path
        
    def load_data(self):
        """Loads data from a CoNLL file.

        Returns:
            list: List of sentences in the dataset.
        """
        with open(self._file_path) as f:

            data = []
            content = f.read()
            docs = [i.strip() for i in content.strip().split('-DOCSTART- -X- -X- O') if i != '']
            for doc in docs:

                #  file content to sentence split
                sentences = doc.strip().split('\n\n')

                if sentences == ['']:
                    data.append(('', ['']))
                    continue

                for sent in sentences:
                    sentence_data = []
                    label_data = []

                    # sentence string to token level split
                    tokens = sent.strip().split('\n')

                    # get annotations from token level split
                    token_list = [t.split() for t in tokens]

                    #  get token and labels from the split
                    for split in token_list:
                        sentence_data.append(split[0])
                        label_data.append((split[-1]))

                    data.append([" ".join(sentence_data), label_data])
      
        data_df = pd.DataFrame(data)
        data_df = data_df.rename(columns={0: "text", 1: "label"})

        return data_df


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


class CSVDataset(_IDataset):
    """Class to handle CSV files dataset. Subclass of _IDataset.
    """

    def __init__(self, file_path) -> None:
        """Initializes CSVDataset object.

        Args:
            file_path (str): Path to the data file.
        """
        super().__init__()
        self._file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """Loads data from a csv file.

        Returns:
            pd.DataFrame: Csv file as a pandas dataframe.
        """
        return pd.read_csv(self._file_path)

