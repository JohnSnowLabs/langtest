class ErrorsWithCodes(type):
    """
    Metaclass that allows associating error/warning codes with messages.

    This metaclass is used to create error and warning classes that provide error codes for easier identification
    and retrieval of error/warning messages.

    Args:
        type (type): The metaclass to inherit from.

    Methods:
        __getattribute__(self, code): Retrieve the error/warning message associated with a given code.

    Usage:
        This metaclass is used to create error and warning classes, such as Errors and Warnings.
    """

    def __getattribute__(self, code):
        """
        Retrieve the error/warning message associated with a given code.

        Args:
            code (str): The error/warning code to look up.

        Returns:
            str: The formatted error/warning message.

        Example:
            error_message = Errors.E000
        """
        msg = super().__getattribute__(code)
        if code.startswith("__"):
            return msg
        else:
            return "[{code}] {msg}".format(code=code, msg=msg)


class Warnings(metaclass=ErrorsWithCodes):
    """
    Class for defining warning messages and associating them with codes.

    This class allows you to define and associate warning messages with unique codes for easier identification and retrieval.

    Usage:
        Create instances of this class to define custom warning messages.

    Example:
        class Warnings(metaclass=ErrorsWithCodes):
            W000 = ("Please run `Harness.run()` before calling `.generated_results()`.")
            W001 = ("No configuration file was provided, loading default config.")
            # ...
    """

    W000 = ("Please run `Harness.run()` before calling `.generated_results()`.")
    W001 = ("No configuration file was provided, loading default config.")
    W002 = ("Default dataset '{info}' successfully loaded.")
    W003 = "Filtering provided bias tests from {len_bias_data} samples - {len_samples_removed} samples removed "
    W004 = (
        "\n"
        + "=" * 100
        + "\nInvalid tokens found in sentence:\n {sent}. \nSkipping sentence.\n"
        + "=" * 100
        + "\n"
    )
    W005 = ("Skipping row {idx} due to invalid data: {row_data} - Exception: {e}")
    W006 = ("target_column '{target_column}' not found in the dataset.")
    W007 = ("'feature_column' '{passage_column}' not found in the dataset.")
    W008 = ("Invalid or Missing label entries in the sentence: {sent}")
    W009 = ("Removing samples where no transformation has been applied:\n")
    W010 = ("- Test '{test}': {count} samples removed out of {total_sample}\n")
    W011 = ("{class_name} successfully ran!")


class Errors(metaclass=ErrorsWithCodes):
    """
    Class for defining error messages and associating them with codes.

    This class allows you to define and associate error messages with unique codes for easier identification and retrieval.

    Usage:
        Create instances of this class to define custom error messages.

    Example:
        class Errors(metaclass=ErrorsWithCodes):
            E000 = ("Each item in the list must be a dictionary")
            E001 = ("Each dictionary in the list must have 'model' and 'hub' keys")
            # ...
    """

    E000 = ("Each item in the list must be a dictionary")
    E001 = ("Each dictionary in the list must have 'model' and 'hub' keys")
    E002 = ("The dictionary must have 'model' and 'hub' keys")
    E003 = ("Invalid 'model' parameter type")
    E004 = (
        "You haven't specified any value for the parameter 'data' and the configuration you "
        "passed is not among the default ones. You need to either specify the parameter 'data' "
        "or use a default configuration."
    )
    E005 = ("Please call .configure() first.")
    E006 = ("Testcases are already generated, please call .run() and .report() next.")
    E007 = ("Bias tests are not applicable for {data_source} dataset.")
    E008 = ("{test_name} tests are not applicable for {data_source} dataset. Please use one of the following datasets: {selected_data_sources}")
    E009 = ("Invalid test: {test_name}")
    E010 = (
        "The test casess have not been generated yet. Please use the `.generate()` method before"
        "calling the `.run()` method."
    )
    E011 = (
        "The tests have not been run yet. Please use the `.run()` method before"
        "calling the `.report()` method."
    )
    E012 = ('You need to set "save_dir" parameter for this format.')
    E013 = ('Report in format "{format}" is not supported. Please use "dataframe", "excel", "html", "markdown", "text", "dict".')
    E014 = ("Custom proportions for {test_name} not found in the test types.")
    E015 = (
        "The current Harness has not been configured yet. Please use the `.configure` method "
        "before calling the `.save` method."
    )
    E016 = (
        "The test cases have not been generated yet. Please use the `.generate` method before"
        "calling the `.save` method."
    )
    E017 = ("File '{filename}' is missing to load a previously saved `Harness`.")
    E018 = ("Unsupported test type '{test_type}'. The available test types are: {available_tests}")
    E019 = ("Invalid 'test_name' value '{test_name}'. It should be one of: Country-Economic-Bias, Religion-Bias, Ethnicity-Name-Bias, Gender-Pronoun-Bias.")
    E020 = ("Invalid 'test_name' value '{test_name}'. It should be one of: Country-Economic-Representation, Religion-Representation, Ethnicity-Representation, Label-Representation.")
    E021 = ("Invalid task type: {category}. Expected 'bias' or 'representation'.")
    E022 = ("A valid token is required for Hugging Face Hub authentication.")
    E023 = ("The '{LIB_NAME}' package is not installed. Please install it using 'pip install {LIB_NAME}'.")
    E024 = ("'file_path' must be a dictionary.")
    E025 = ("The 'data_source' key must be provided in the 'file_path' dictionary.")
    E026 = ("Given task ({task}) is not matched with template. \
            CSV dataset can ne only loaded for text-classification and ner!")
    E027 = ("feature_column '{feature_column}' not found in the dataset.")
    E028 = (
        "CSV file is invalid. CSV handler works with template column names!\n"
        "{not_referenced_columns_keys} column could not be found in header.\n"
        "You can use following namespaces:\n{not_referenced_columns}"
    )
    E029 = (
        "Your dataset needs to have at least have a column with one of the following name: "
        "{valid_column_names}, found: {column_names}."
    )
    E030 = ("Invalid dataset name: {dataset_name}")
    E031 = ("Class '{class_name}Formatter' not yet implemented.")
    E032 = ("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")
    E033 = ("Input arrays must be of type 'np.ndarray', but received types: {type_a} and {type_b}")
    E034 = ("Distance function '{name}' not found")
    E035 = (
        "Input strings must be of type 'str', but received types: {type_a} and {type_b}"
    )
    E036 = ("String distance function '{name}' not found")
    E037 = "Model must have a predict method"
    E038 = (
        "Invalid SparkNLP model object: {model_type}. "
        "John Snow Labs model handler accepts: "
        "[NLUPipeline, PretrainedPipeline, PipelineModel, LightPipeline]"
    )
    E039 = (
        "johnsnowlabs is not installed. "
        "In order to use NLP Models Hub, johnsnowlabs should be installed!"
    )
    E040 = ("Invalid PipelineModel! There should be at least one {var} component.")
    E041 = ("Model '{path}' is not found online or local. Please install it by python -m spacy download {path} or check the path.")
    E042 = ("Provided model hub is not supported. Please choose one of the supported model hubs: {supported_hubs}")
    E043 = ("Provided task is not supported. Please choose one of the supported tasks: {l}")
    E044 = ("Model '{path}' is not found online or local. Please install langchain by pip install langchain")
    E045 = ("Please update model_parameters section in config.yml file for '{path}' model in '{hub}'.\nmodel_parameters:\n\t{field}: value\n\n{required_fields} is/are required field(s), please provide them in config.yml")
    E046 = ('The test type "{sub_test}" is not supported for the task "{task}". "{sub_test}" only supports {supported}.')
    E047 = ("The test category {test_category} does not exist. Available categories are: {available_categories}.")
    E048 = ("Invalid test configuration! Tests can be [1] dictionary of test name and corresponding parameters.")
    E049 = ("Invalid test specification: {not_supported_tests}. Available tests are: {supported_tests}")
    E050 = ("Invalid perturbation {key} in multiple_perturbations. Please use perturbations1, perturbations2, ...")
    E051 = ("multiple_perturbations test is not supported for NER task")
    E052 = ("This dataset does not contain labels, and {var} tests cannot be run with it.")
    E053 = ("The dataset {dataset_name} does not contain labels, and fairness tests cannot be run with it. Skipping the fairness tests.")
    E054 = ("Invalid schema. It should be one of: {var}.")
    E055 = ("Invalid JSON format. 'name' key is missing.")
    E056 = ("Invalid 'name' value '{var1}'. It should be one of: {var2}.")
    E057 = ("Invalid 'name' format in the JSON file.")
    E058 = ("At least one of 'first_names' or 'last_names' must be specified for '{name}'.")
    E059 = ("'last_names' must be specified for '{name}'.")
    E060 = (
        "Invalid keys in the JSON for '{name}'. "
        "Only the following keys are allowed: 'name', 'first_names', 'last_names'."
    )
    E061 = (
        "Missing pronoun keys in the JSON for '{name}'. Please include at least one of: "
        "'subjective_pronouns', 'objective_pronouns', 'reflexive_pronouns', 'possessive_pronouns'."
    )
    E062 = (
        "Invalid keys in the JSON for '{var1}': {var2}. "
        "Only the following keys are allowed: "
        "'name', 'subjective_pronouns', 'objective_pronouns', 'reflexive_pronouns', 'possessive_pronouns'."
    )
    E063 = ("This method must be implemented in the derived class.")
    E064 = ("Sum of proportions cannot be greater than 1. So {var} test cannot run.")
    E065 = ("In order to generate test cases for swap_entities, {var} should be passed!")
    E066 = ("Add context strategy must be one of 'start', 'end', 'combined'. Cannot be {strategy}.")
    E067 = ("Unknown transformation: {order}")
    E068 = ("Invalid JSON format. Format should be a list of strings.")
    E069 = ("pass_custom_data() method with test_name as 'Label-Representation' is only supported for NER task.")
    E070 = ("Unsupported task: '{var}'")
    E071 = (
        "You are trying to access a gated repo. Make sure to request access at "
        "{model_name} and pass a token having permission to this repo either by logging in with `huggingface-cli login` or by setting the `HUGGINGFACEHUB_API_TOKEN` environment variable with your API token."
    )
    E072 = ("Number out of range")
    E073 = (
        "Dictionary Error: Search term or search token not found in dictionary. "
        "Contact administrator to update dictionary if necessary."
    )
    E074 = ("Invalid averaging method. Must be one of 'macro', 'micro', or 'weighted'.")
    E075 = ("No {hub_name} embeddings class found")
    E076 = ("Unsupported {metric} distance metric: {selected_metric}")
    E077 = ("\nThe provided columns are not supported for creating a sample.\
            \nPlease choose one of the supported columns: {supported_columns}\
            \nOr classify the features and target columns from the {given_columns}")


class ColumnNameError(Exception):
    """ColumnNameError class is used to raise an exception
    when the column name is not found in the dataset."""

    def __init__(
        self,
        supported_columns,
        given_columns,
    ):
        self.message = Errors.E077.format(
            supported_columns=supported_columns, given_columns=given_columns
        )
        super().__init__(self.message)
