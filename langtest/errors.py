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

    W000 = "Please run `Harness.run()` before calling `.generated_results()`."
    W001 = "No configuration file was provided, loading default config."
    W002 = "Default dataset '{info}' successfully loaded."
    W003 = "Filtering provided bias tests from {len_bias_data} samples - {len_samples_removed} samples removed "
    W004 = (
        "\n"
        + "=" * 100
        + "\nInvalid tokens found in sentence:\n {sent}. \nSkipping sentence.\n"
        + "=" * 100
        + "\n"
    )
    W005 = "Skipping row {idx} due to invalid data: {e}"
    W006 = "target_column '{target_column}' not found in the dataset."
    W007 = "'feature_column' '{passage_column}' not found in the dataset."
    W008 = "Invalid or Missing label entries in the sentence: {sent}"
    W009 = "Removing samples where no transformation has been applied:\n"
    W010 = "- Test '{test}': {count} samples removed out of {total_sample}\n"
    W011 = "{class_name} successfully ran!"


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

    E000 = "Each item in the list must be a dictionary"
    E001 = "Each dictionary in the list must have 'model' and 'hub' keys"
    E002 = "The dictionary must have 'model' and 'hub' keys"
    E003 = "Invalid 'model' parameter type"
    E004 = (
        "You haven't specified any value for the parameter 'data' and the configuration you "
        "passed is not among the default ones. You need to either specify the parameter 'data' "
        "or use a default configuration."
    )
