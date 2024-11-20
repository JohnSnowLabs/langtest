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


class RobustnessTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Robustness Tests.
    """

    from langtest.transform import robustness

    uppercase: robustness.UpperCase.TestConfig
    lowercase: robustness.LowerCase.TestConfig
    titlecase: robustness.TitleCase.TestConfig
    add_punctuation: robustness.AddPunctuation.TestConfig
    strip_punctuation: robustness.StripPunctuation.TestConfig
    add_typo: robustness.AddTypo.TestConfig
    swap_entities: robustness.SwapEntities.TestConfig
    american_to_british: robustness.ConvertAccent.TestConfig
    british_to_american: robustness.ConvertAccent.TestConfig
    add_context: robustness.AddContext.TestConfig
    add_contractions: robustness.AddContraction.TestConfig
    dyslexia_word_swap: robustness.DyslexiaWordSwap.TestConfig
    number_to_word: robustness.NumberToWord.TestConfig
    add_ocr_typo: robustness.AddOcrTypo.TestConfig
    add_abbreviation: robustness.AbbreviationInsertion.TestConfig
    add_speech_to_text_typo: robustness.AddSpeechToTextTypo.TestConfig
    add_slangs: robustness.AddSlangifyTypo.TestConfig
    multiple_perturbations: robustness.MultiplePerturbations.TestConfig
    adjective_synonym_swap: robustness.AdjectiveSynonymSwap.TestConfig
    adjective_antonym_swap: robustness.AdjectiveAntonymSwap.TestConfig
    strip_all_punctuation: robustness.StripAllPunctuation.TestConfig
    randomize_age: robustness.RandomAge.TestConfig
    add_new_lines: robustness.AddNewLines.TestConfig
    add_tabs: robustness.AddTabs.TestConfig
