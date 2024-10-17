from typing import Literal, TypedDict, Union, List


class ModelConfig(TypedDict):
    """
    ModelConfig is a TypedDict that defines the configuration for a model.

    Attributes:
        model (str): The name of the model.
        type (Literal['chat', 'completion']): The type of the model, either 'chat' or 'completion'.
        hub (str): The hub where the model is located.
    """

    model: str
    type: Literal["chat", "completion"]
    hub: str


class DatasetConfig(TypedDict):
    """
    DatasetConfig is a TypedDict that defines the configuration for a dataset.

    Attributes:
        data_source (str): The source of the data, e.g., a file path.
        split (str): The data split, e.g., 'train', 'test', or 'validation'.
        subset (str): A specific subset of the data, if applicable.
        feature_column (Union[str, List[str]]): The column(s) representing the features in the dataset.
        target_column (Union[str, List[str]]): The column(s) representing the target variable(s) in the dataset.
        source (str): The original source of the dataset ex: huggingface.
    """

    data_source: str
    split: str
    subset: str
    feature_column: Union[str, List[str]]
    target_column: Union[str, List[str]]
    source: str
