from typing import Any, Dict, Literal, Optional, TypedDict, Union, List


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


class BiasTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Bias Tests.
    """

    from langtest.transform import bias

    replace_to_male_pronouns: bias.GenderPronounBias.TestConfig
    replace_to_female_pronouns: bias.GenderPronounBias.TestConfig
    replace_to_neutral_pronouns: bias.GenderPronounBias.TestConfig
    replace_to_high_income_country: bias.CountryEconomicBias.TestConfig
    replace_to_low_income_country: bias.CountryEconomicBias.TestConfig
    replace_to_upper_middle_income_country: bias.CountryEconomicBias.TestConfig
    replace_to_lower_middle_income_country: bias.CountryEconomicBias.TestConfig
    replace_to_white_firstnames: bias.EthnicityNameBias.TestConfig
    replace_to_black_firstnames: bias.EthnicityNameBias.TestConfig
    replace_to_hispanic_firstnames: bias.EthnicityNameBias.TestConfig
    replace_to_asian_firstnames: bias.EthnicityNameBias.TestConfig
    replace_to_white_lastnames: bias.EthnicityNameBias.TestConfig
    replace_to_black_lastnames: bias.EthnicityNameBias.TestConfig
    replace_to_hispanic_lastnames: bias.EthnicityNameBias.TestConfig
    replace_to_asian_lastnames: bias.EthnicityNameBias.TestConfig
    replace_to_native_american_lastnames: bias.EthnicityNameBias.TestConfig
    replace_to_inter_racial_lastnames: bias.EthnicityNameBias.TestConfig
    replace_to_muslim_names: bias.ReligionBias.TestConfig
    replace_to_hindu_names: bias.ReligionBias.TestConfig
    replace_to_christian_names: bias.ReligionBias.TestConfig
    replace_to_sikh_names: bias.ReligionBias.TestConfig
    replace_to_jain_names: bias.ReligionBias.TestConfig
    replace_to_parsi_names: bias.ReligionBias.TestConfig
    replace_to_buddhist_names: bias.ReligionBias.TestConfig


class RepresentationTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Representation Tests.
    """

    from langtest.transform import representation

    min_gender_representation_count: representation.GenderRepresentation.TestConfig
    min_gender_representation_proportion: representation.GenderRepresentation.TestConfig
    min_ethnicity_name_representation_count: (
        representation.EthnicityRepresentation.TestConfig
    )
    min_ethnicity_name_representation_proportion: (
        representation.EthnicityRepresentation.TestConfig
    )
    min_label_representation_count: representation.LabelRepresentation.TestConfig
    min_label_representation_proportion: representation.LabelRepresentation.TestConfig
    min_religion_name_representation_count: (
        representation.ReligionRepresentation.TestConfig
    )
    min_religion_name_representation_proportion: (
        representation.ReligionRepresentation.TestConfig
    )
    min_country_economic_representation_count: (
        representation.CountryEconomicRepresentation.TestConfig
    )
    min_country_economic_representation_proportion: (
        representation.CountryEconomicRepresentation.TestConfig
    )


class FairnessTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Fairness Tests.
    """

    from langtest.transform import fairness

    min_gender_f1_score: fairness.MinGenderF1Score.TestConfig
    max_gender_f1_score: fairness.MaxGenderF1Score.TestConfig
    min_gender_rouge1_score: fairness.MinGenderRougeScore.TestConfig
    min_gender_rouge2_score: fairness.MinGenderRougeScore.TestConfig
    min_gender_rougeL_score: fairness.MinGenderRougeScore.TestConfig
    min_gender_rougeLsum_score: fairness.MinGenderRougeScore.TestConfig
    max_gender_rouge1_score: fairness.MaxGenderRougeScore.TestConfig
    max_gender_rouge2_score: fairness.MaxGenderRougeScore.TestConfig
    max_gender_rougeL_score: fairness.MaxGenderRougeScore.TestConfig
    max_gender_rougeLsum_score: fairness.MaxGenderRougeScore.TestConfig
    min_gender_llm_eval: fairness.MinGenderLLMEval.TestConfig
    max_gender_llm_eval: fairness.MaxGenderLLMEval.TestConfig


class AccuracyTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Accuracy Tests.
    """

    from langtest.transform import accuracy

    min_precision_score: accuracy.MinPrecisionScore.TestConfig
    min_recall_score: accuracy.MinRecallScore.TestConfig
    min_f1_score: accuracy.MinF1Score.TestConfig
    min_micro_f1_score: accuracy.MinMicroF1Score.TestConfig
    min_macro_f1_score: accuracy.MinMacroF1Score.TestConfig
    min_weighted_f1_score: accuracy.MinWeightedF1Score.TestConfig
    min_exact_match_score: accuracy.MinEMcore.TestConfig
    min_bleu_score: accuracy.MinROUGEcore.TestConfig
    min_rouge1_score: accuracy.MinROUGEcore.TestConfig
    min_rouge2_score: accuracy.MinROUGEcore.TestConfig
    min_rougeL_score: accuracy.MinROUGEcore.TestConfig
    min_rougeLsum_score: accuracy.MinROUGEcore.TestConfig
    llm_eval: accuracy.LLMEval.TestConfig
    degradation_analysis: accuracy.DegradationAnalysis.TestConfig


class ToxicityTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Toxicity Tests.
    """

    from langtest.transform import toxicity

    general_toxicity: toxicity.GeneralToxicity.TestConfig
    obscene: toxicity.ToxicityTypes.TestConfig
    insult: toxicity.ToxicityTypes.TestConfig
    threat: toxicity.ToxicityTypes.TestConfig
    identity_attack: toxicity.ToxicityTypes.TestConfig
    homosexual_gay_or_lesbian: toxicity.ToxicityTypes.TestConfig


class SecurityTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Security Tests.
    """

    from langtest.transform import security

    prompt_injection_attack: security.PromptInjection.TestConfig


class SafetyTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Safety Tests.
    """

    from langtest.transform import safety

    misuse: safety.Misuse.TestConfig
    injection_probalities_score: safety.InjectionProbalities.TestConfig
    jailbreak_probalities_score: safety.JailBreakProbalities.TestConfig


class PerformanceTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Performance Tests.
    """

    from langtest.transform import performance

    speed: performance.Speed.TestConfig


class LegalTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Legal Tests.
    """

    from langtest.transform import legal

    legal_support: legal.LegalTestFactory.TestConfig


class GrammarTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Grammar Tests.
    """

    from langtest.transform import grammar

    paraphase: grammar.Paraphrase.TestConfig


class ClinicalTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Clinical Tests.
    """

    from langtest.transform import clinical

    demographic_bias: clinical.DemographicBias.TestConfig
    drug_generic_to_brand: clinical.Generic2Brand.TestConfig
    drug_brand_to_generic: clinical.Brand2Generic.TestConfig


class SensitivityTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Sensitivity Tests.
    """

    from langtest.transform import sensitivity

    add_negation: sensitivity.AddNegation.TestConfig
    add_toxic_words: sensitivity.AddToxicWords.TestConfig


class SterotypeTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Sterotype Tests.
    """

    from langtest.transform import stereotype

    crows_pairs: stereotype.StereoTypeTestFactory.TestConfig
    wino_bias: stereotype.StereoTypeTestFactory.TestConfig


class SterosetTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Steroset Tests.
    """

    from langtest.transform import stereoset

    intrasentence: stereoset.StereoSetTestFactory.TestConfig
    intersentence: stereoset.StereoSetTestFactory.TestConfig


class SycophancyTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Sycophancy Tests.
    """

    from langtest.transform import sycophancy

    sycophancy_math: sycophancy.SycophancyMath.TestConfig
    sycophancy_nlp: sycophancy.SycophancyNlp.TestConfig


class DisinformationTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Disinformation Tests.
    """

    from langtest.transform import disinformation

    narrative_wedging: disinformation.DisinformationTestFactory.TestConfig


class IdeologyTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Ideology Tests.
    """

    from langtest.transform import ideology

    political_compass: Optional[ideology.PoliticalCompass.TestConfig]


class FactualityTestsConfig(TypedDict):
    """
    TestsConfig is for defining the configuration of a Factuality Tests.
    """

    from langtest.transform import factuality

    order_bias: factuality.FactualityTestFactory.TestConfig


class TestCategories(TypedDict):
    """
    TestCategories is a TypedDict that defines the categories of tests.

    """

    robustness: RobustnessTestsConfig
    bias: BiasTestsConfig
    representation: RepresentationTestsConfig
    fairness: FairnessTestsConfig
    accuracy: AccuracyTestsConfig
    toxicity: ToxicityTestsConfig
    security: SecurityTestsConfig
    safety: SafetyTestsConfig
    performance: PerformanceTestsConfig
    legal: LegalTestsConfig
    grammar: GrammarTestsConfig
    clinical: ClinicalTestsConfig
    sensitivity: SensitivityTestsConfig
    stereotype: SterotypeTestsConfig
    stereoset: SterosetTestsConfig
    sycophancy: SycophancyTestsConfig
    disinformation: DisinformationTestsConfig
    ideology: IdeologyTestsConfig
    factuality: FactualityTestsConfig


class HarnessConfig(TypedDict):
    """
    HarnessConfig is a TypedDict that defines the configuration for a harness.
    """

    evaluation: Dict[str, Any]
    model_parameters: Dict[str, Any]
    tests: TestCategories
