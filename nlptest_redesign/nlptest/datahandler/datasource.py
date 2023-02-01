from abc import ABC, abstractmethod 
import pandas as pd
import os 

class _IDataset(ABC):

    @abstractmethod
    def load_data(self):
        return NotImplemented


class DataFactory:

    def __init__(self, file_path) -> None:
        self._file_path = file_path
        self._class_map = {cls.__name__.replace('Dataset', '').lower(): cls for cls in _IDataset.__subclasses__()}
        _, self.file_ext = os.path.splitext(self._file_path)

    def load(self):
        return self._class_map[self.file_ext.replace('.', '')](self._file_path).load_data()


class ConllDataset(_IDataset):

    def __init__(self, file_path) -> None:
        super().__init__()
        self._file_path = file_path
        
    def load_data(self):
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

    def __init__(self, file_path) -> None:
        super().__init__()
        self._file_path = file_path

    def load_data(self):
        pass


class CSVDataset(_IDataset):

    def __init__(self, file_path) -> None:
        super().__init__()
        self._file_path = file_path

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self._file_path)

