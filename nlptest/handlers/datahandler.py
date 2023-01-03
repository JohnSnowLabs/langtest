"""This module contains classes for storing and manipulating data for named entity recognition (NER).

The NERSample class represents a single token in a sentence with its corresponding label, pos tag, and chunk tag.
The NERDataHandler class stores and provides functions for working with a collection of NERSample instances.

Example:

    data_handler = NERDataHandler.load_conll('path/to/file.conll')

    #   filter data containing specific named entity types.
    filtered_data_handler = data_handler.filter_by_entity_type('PER')

    #   set spark to use SparkNLP backend
    data_handler.set_spark(spark_session)

    #   get subset of data containing specific sentence indices.
    sample_data_handler = data_handler[[12, 17, 19, 20, 24, 45]]

    #   get sample from the data handler.
    sample_data_handler = data_handler.get_sample(10, random_seed=42)
"""

import random
import numpy as np
from typing import Optional, List, Union, Tuple, Iterable, Dict
from copy import deepcopy

#   SPARK IMPORT SHOULD BE CHECKED
from pyspark.sql.types import *
from pyspark.sql.types import Row
from pyspark.sql import SparkSession, DataFrame

_ROW_SCHEMA = ArrayType(StructType([
    StructField("annotatorType", StringType()),
    StructField("begin", IntegerType()),
    StructField("end", IntegerType()),
    StructField("result", StringType()),
    StructField("metadata", MapType(StringType(), StringType())),
    StructField("embeddings", ArrayType(FloatType())),
]))

_DF_SCHEMA = StructType([
    StructField("text", StringType()),
    StructField("document", _ROW_SCHEMA),
    StructField("sentence", _ROW_SCHEMA),
    StructField("token", _ROW_SCHEMA),
    StructField("pos", _ROW_SCHEMA),
    StructField("label", _ROW_SCHEMA)
])


class NERSample:
    """Class for storing information about a named entity recognition (NER) token samples.

    A NER sample consists of a chunk of entity, a label indicating the type of named
    entity, and a list of tokens and optional POS tags and chunk tags. The sample
    may also have an ignore flag indicating that it should be ignored in evaluation.
    """

    def __init__(
            self,
            chunk: str,
            label: str,
            tokens: List[str],
            pos_tags: Optional[List[str]] = None,
            chunk_tags: Optional[List[str]] = None,
    ):
        """Initialize a NERSample instance.

        Args:
            chunk: A string representing the chunk of text.
            label: A string representing the label of the named entity.
            tokens: A list of strings representing the tokens in the chunk.
            pos_tags: (Optional) A list of strings representing the POS tags of the
                tokens. Defaults to None.
            chunk_tags: (Optional) A list of strings representing the chunk tags of
                the tokens. Defaults to None.

        """
        self.chunk = chunk
        self.label = label
        self.tokens = tokens
        self.pos_tags = pos_tags
        self.chunk_tags = chunk_tags
        self.is_ignore = False

    def ignore(self):
        """Set the ignore flag for the sample to True."""
        self.is_ignore = True

    def __repr__(self):
        """Return a string representation of the NERSample instance."""
        return f"({', '.join([key + ': ' + str(value) for key, value in self.__dict__.items()])})"

    def __len__(self):
        """Return the length of the tokens list for the sample."""
        return len(self.tokens)

    def __iter__(self):
        """Yield the ner sample with"""
        chunk_tokens = self.chunk.split()
        chunk_labels = ['B-' + self.label]
        chunk_labels.extend(['I-' + self.label] * (len(self) - 1))
        for i in range(len(self)):
            token_level_info = (chunk_tokens[i], chunk_labels[i])
            if self.pos_tags:
                token_level_info += (self.pos_tags[i],)
            if self.chunk_tags:
                token_level_info += (self.chunk_tags[i],)
            yield token_level_info


class NERDataHandler:
    """Class for handling named entity recognition (NER) data for different data formats.

    The NERDataHandler class designed to store different NER data formats. It contains
    list of NERSample instances, along with indices for sentences and documents. First
    document string of the file is accepted as doc_string and used to split multiple
    documents while saving data to CoNLL file. If not given; unique_entities, entity
    instances is created automatically. Given unique entities should be ordered according to
    their label ids; however, label ids can be set later on with `set_label_ids` method.
    Optionally, SparkSession instance can be passed during the initialization or after the
    initialization via `set_spark` method.
    """

    def __init__(
            self,
            data_container: List[List[NERSample]],
            sent_indices: Optional[List[int]] = None,
            doc_indices: Optional[List[int]] = None,
            doc_string: Optional[str] = None,
            unique_entities: Optional[str] = None,
            entity_instances_mx: np.array = None,
            spark_session: Optional[SparkSession] = None,
    ):
        """Initialize a NERDataHandler instance manually. It is suggested to use load
        methods e.g `load_conll` to create class instance automatically.

        Args:
            data_container: A list of lists of NERSample instances representing the
                data.
            sent_indices: (Optional) A list of integers representing the indices of
                the sentences in the data. Defaults, created automatically via load methods.
            doc_indices: (Optional) A list of integers representing the indices of
                the documents in the data. Defaults, created automatically via load methods.
            doc_string: (Optional) A string representing the raw document string. Should be
                started with "-DOCSTART-". Defaults, created automatically via load methods.
            unique_entities: (Optional) A list of strings representing the unique
                entity labels in the data. Defaults, created automatically via load methods.
            entity_instances_mx: (Optional) A numpy array representing the matrix of
                entity instances. Defaults, created automatically via load methods.
            spark_session: (Optional) A SparkSession instance. Defaults to None. Can be set after
                via `set_spark` method. Required to use SparkNLP backend.
        """

        self.data_container = data_container
        self.doc_indices = doc_indices
        self.doc_string = doc_string

        if sent_indices is None:
            sent_indices = list(range(len(self.data_container)))
        self.sent_indices = sent_indices

        if unique_entities is None:
            unique_entities = self.__get_unique_entities()
        self.entities = unique_entities

        self.label2id = {label: i for i, label in enumerate(self.entities)}
        self.id2label = {i: label for i, label in enumerate(self.entities)}

        if entity_instances_mx is None:
            entity_instances_mx = self.__create_ent_instance_mx()
        self.entity_instances = entity_instances_mx

        self.spark_session = spark_session

    @classmethod
    def load_conll(cls, filepath: str) -> 'NERDataHandler':
        """Load data from a file in CoNLL format.

        Args:
            filepath: A string representing the file path of the CoNLL file.

        Returns:
            A NERDataHandler instance initialized with the data from the CoNLL file.
        """
        doc_indices = []
        doc_string = None
        sentence_count = 0
        sentence_samples = []
        curr_sentence = []

        curr_chunk = []
        curr_label = []
        curr_pos_tags = []
        curr_chunk_tags = []

        with open(filepath, 'r') as conll:

            for line in conll:

                #   sentence separator line
                if line is '\n':

                    if curr_sentence:
                        sentence_samples.append(curr_sentence)
                        sentence_count += 1

                    curr_sentence = []
                    continue

                #   document string line
                if line.startswith('-DOCSTART-'):
                    doc_indices.append(sentence_count)
                    #   first document string accepted as doc_string
                    if doc_string is None:
                        doc_string = line.strip()
                    continue

                #   token-level information line
                token, pos_tag, chunk_tag, label = line.strip().split()

                if label[0] == 'B':

                    if curr_chunk:
                        curr_sentence.append(
                            NERSample(
                                chunk=' '.join(curr_chunk),
                                label=curr_label[0],
                                tokens=curr_chunk,
                                pos_tags=curr_pos_tags,
                                chunk_tags=curr_chunk_tags
                            )
                        )

                        curr_chunk = []
                        curr_label = []
                        curr_pos_tags = []
                        curr_chunk_tags = []

                    curr_chunk.append(token)
                    curr_label.append(label[2:])
                    curr_pos_tags.append(pos_tag)
                    curr_chunk_tags.append(chunk_tag)

                elif label[0] == 'I':
                    curr_chunk.append(token)
                    curr_pos_tags.append(pos_tag)
                    curr_chunk_tags.append(chunk_tag)

                else:

                    if curr_chunk:
                        curr_sentence.append(
                            NERSample(
                                chunk=' '.join(curr_chunk),
                                label=curr_label[0],
                                tokens=curr_chunk,
                                pos_tags=curr_pos_tags,
                                chunk_tags=curr_chunk_tags
                            )
                        )
                        curr_chunk = []
                        curr_label = []
                        curr_pos_tags = []
                        curr_chunk_tags = []

                    curr_sentence.append(
                        NERSample(
                            chunk=token,
                            label=label,
                            tokens=[token],
                            pos_tags=[pos_tag],
                            chunk_tags=[chunk_tag]
                        )
                    )

        if curr_sentence:
            sentence_samples.append(curr_sentence)

        return cls(
            data_container=sentence_samples,
            doc_indices=doc_indices,
            doc_string=doc_string,
        )

    def __get_unique_entities(self) -> List[str]:
        """Extract unique entities from data container."""
        unique_entities = []
        for sent_samples in self.data_container:
            for ner_sample in sent_samples:
                if ner_sample.label not in unique_entities:
                    unique_entities.append(ner_sample.label)
        return unique_entities

    def __create_ent_instance_mx(self) -> np.array:
        """Create entity instance matrix to optimize filtering and keeping track of entity instances."""
        entity_instances = np.zeros((len(self), len(self.entities)), dtype=int)
        for sent_indx, sent_samples in enumerate(self.data_container):
            for ner_sample in sent_samples:
                token_indx = self.label2id[ner_sample.label]
                entity_instances[sent_indx][token_indx] += 1

        return entity_instances

    def write_conll(self, filepath: str) -> None:
        """Write the data to a file in CoNLL format.

        Args:
            filepath: The file path to write the data to.

        Returns:
            None
        """
        try:
            with open(filepath, 'w') as f:
                try:
                    counter = 0
                    if self.doc_string:
                        f.write(f"{self.doc_string}\n")
                        counter += 1

                    for indx, sentence_samples in enumerate(self.data_container):

                        if counter < len(self.doc_indices) and indx == self.doc_indices[counter]:
                            f.write(f"\n{self.doc_string}\n")
                            counter += 1

                        f.write("\n")
                        for ner_sample in sentence_samples:
                            for token, label, pos_tag, chunk_tag in ner_sample:
                                f.write(f"{token} {pos_tag} {chunk_tag} {label}\n")

                except (IOError, OSError) as e:
                    print(f"Error while writing to the {filepath}.")
                    print(e)

        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Error while opening the {filepath}.")
            print(e)

    def get_sample(self, k: int, random_seed: int = None) -> 'NERDataHandler':
        """Get random k sample from the data.

        Args:
            k (int): Number of samples to return.
            random_seed (int): Optional integer seed to initialize the pseudorandom.

        Returns:
            A NERDataHandler instance initialized with the sampled data from the current instance.
        """
        if random_seed is not None:
            random.seed(random_seed)

        sample_sent_indices = random.sample(self.sent_indices, k)

        return self[sample_sent_indices]

    def filter_by_entity_type(self, entity_type: Union[List[str], str]) -> 'NERDataHandler':
        """Filter data handler with the given entity type.

        Args:
            entity_type (List[str], str): Desired entity type to filter data.

        Returns:
            A NERDataHandler instance initialized with the filtered data from the current instance.
        """
        if isinstance(entity_type, str):
            entity_type = [entity_type]

        for ent in entity_type:
            if ent not in self.entities:
                raise ValueError(f"entity_type: {ent} is invalid! Existing entities are: {list(self.label2id.keys())}")

        entity_indices = [self.label2id[label] for label in entity_type]
        filtered_entity_mx = np.take(self.entity_instances, entity_indices, axis=1)
        filter_indx, = np.where(np.all(filtered_entity_mx == entity_indices, axis=1))

        return self[filter_indx]

    def get_label_counts(self, entity_type: Optional[Union[List[str], str]] = None) -> Dict[str, int]:
        """Create dictionary of entities indicating their number of instances in the data.

        Args:
            entity_type (List[str], str): Desired entity type to count instances.

        Returns:
            Dictionary of entities and their number of instances.
        """
        if entity_type is None:
            entity_type = self.entities

        for ent in entity_type:
            if ent not in self.entities:
                raise ValueError(f"entity_type: {ent} is invalid! Existing entities are: {list(self.label2id.keys())}")

        entity_indices = [self.label2id[label] for label in entity_type]

        count_dict = {}
        for i, ent in enumerate(entity_type):
            total_ent_count = self.entity_instances[:, entity_indices[i]].sum()
            count_dict[ent] = total_ent_count

        return count_dict

    @staticmethod
    def samples_to_spark_rows(list_of_samples: List[NERSample]):
        """Helper method to create Spark rows for each NER sentence sample."""

        sentence = " ".join([" ".join(sample.tokens) for sample in list_of_samples])

        token_rows = []
        label_rows = []
        pos_tag_rows = []

        begin = end = 0
        for ner_sample in list_of_samples:

            iob_labels = ['O'] * len(ner_sample)
            if ner_sample.label != 'O':
                iob_labels[0] = 'B-' + ner_sample.label
                for i in range(len(ner_sample) - 1):
                    iob_labels[1 + i] = 'I-' + ner_sample.label

            for token, label, pos_tag in zip(ner_sample.tokens, iob_labels, ner_sample.pos_tags):
                end += len(token) - 1

                token_rows.append(Row(
                    annotatorType='token',
                    begin=begin,
                    end=end,
                    result=token,
                    metadata={'sentence': '0'},
                    embeddings=[]
                )),

                label_rows.append(Row(
                    annotatorType='named_entity',
                    begin=begin,
                    end=end,
                    result=label,
                    metadata={'sentence': '0', 'word': token},
                    embeddings=[]
                ))

                pos_tag_rows.append(Row(
                    annotatorType='pos',
                    begin=begin,
                    end=end,
                    result=pos_tag,
                    metadata={'sentence': '0', 'word': token},
                    embeddings=[]
                ))

                end = begin = end + 2

        return {
            'text': sentence,
            'document': [Row(
                annotatorType='document',
                begin=0,
                end=len(sentence) - 1,
                result=sentence,
                metadata={'training': 'true'},
                embeddings=[]
            )],
            'sentence': [Row(
                annotatorType='document',
                begin=0,
                end=len(sentence) - 1,
                result=sentence,
                metadata={'sentence': '0'},
                embeddings=[]
            )],
            'pos': pos_tag_rows,
            'token': token_rows,
            'label': label_rows
        }

    def to_spark_dataframe(self) -> DataFrame:
        """Create a Spark DataFrame from the data and labels in the NERDataHandler instance.

        Returns:
            A Spark DataFrame with columns 'text', 'document', 'sentence', 'token', 'label', 'pos'.

        Raises:
            ValueError: If the Spark session has not been initialized.
        """
        if self.spark_session is None:
            raise ValueError(
                'Spark session is not initialized! Set spark session by `data_handler.set_spark(spark_session)`.')

        spark_data = []
        for sentence_sample in self.data_container:
            sentence_rows = self.samples_to_spark_rows(sentence_sample)
            spark_data.append(
                sentence_rows
            )

        return self.spark_session.createDataFrame(spark_data, _DF_SCHEMA)

    def set_spark(self, spark_session: SparkSession) -> SparkSession:
        """Set the Spark session for the NERDataHandler instance.

        Args:
            spark_session: A SparkSession instance.

        Returns:
            Given Spark Session instance, prints information about the spark session.
        """
        self.spark_session = spark_session
        return spark_session

    def copy(self):
        """Create a deep copy of the NERDataHandler instance.

        Returns:
            A deep copy of the NERDataHandler instance.
        """
        return deepcopy(self)

    #   Special Methods
    def __iter__(self) -> List[NERSample]:
        """Iterate through the NERSample instances in the data container.

        Yields:
            A list of NERSample instances representing a sentence.
        """
        for sentence_samples in self.data_container:
            yield sentence_samples

    def __getitem__(self, sample_indices: Union[Iterable[int], slice, int]) -> 'NERDataHandler':
        """Get a subset of the data based on the indices of the sentences. Sample indices can
        be passed as single int or list of integers to get the subset. Sample indices should be
        in `self.sent_indices`.

        Args:
            sample_indices: An integer or iterable of integers representing sent_indices
            to include in the subset.

        Returns:
            A NERDataHandler instance containing the subset of the data.

        Raises:
            IndexError: If any of the provided sentence indices are invalid.
        """
        if isinstance(sample_indices, int):
            sample_indices = [sample_indices]

        if isinstance(sample_indices, slice):
            sample_indices = self.sent_indices[sample_indices]

        sample_indices = np.unique(np.fromiter(sample_indices, np.int))
        data_indices, = np.where(np.isin(self.sent_indices, sample_indices))

        if sample_indices.shape != data_indices.shape:
            invalid_indices, = np.where(~np.isin(sample_indices, self.sent_indices, assume_unique=True))
            raise IndexError(f'sent_indices are not found! Invalid indices: {sample_indices[invalid_indices]}')

        data_container = [self.data_container[i] for i in data_indices]
        sample_entity_instances_mx = np.take(self.entity_instances, data_indices, axis=0)

        return self.__class__(
            data_container=data_container,
            sent_indices=sample_indices,
            doc_indices=self.doc_indices,
            doc_string=self.doc_string,
            unique_entities=self.entities,
            entity_instances_mx=sample_entity_instances_mx,
            spark_session=self.spark_session
        )

    def __setitem__(self, indx: Tuple[int, int], ner_sample):
        """Set the NERSample at a specific (sentence index, token index).

        Args:
            indx: A tuple of (sentence index, token index) representing the position of the NERSample.
            ner_sample: The NERSample to set at the specified index.

        Raises:
            IndexError: If the sentence index is not found or if the token index is out of range in the sentence.
        """
        sent_indx, token_indx = indx
        data_indx, = np.where(self.sent_indices == sent_indx)

        if data_indx.size == 0:
            raise IndexError(f'sentence index is not found! Invalid index: {sent_indx}')
        data_indx = data_indx[0]

        num_token = len(self.data_container[data_indx])
        if num_token <= token_indx:
            raise IndexError(
                f"Index out of range! Number of token in sample is '{num_token}', "
                f"but given token index is '{token_indx}'.")

        self.data_container[data_indx][token_indx] = ner_sample

    def __len__(self) -> int:
        """Get the number of sentences in the data container."""
        return len(self.data_container)