import importlib
import logging


def try_import_lib(lib: str) -> bool:
    """Tries to import a Python library/module dynamically and returns True if successful, False otherwise.

    Args:
        lib (str): The name of the library/module to import.

    Returns:
        bool: True if the import is successful, False otherwise.
    """
    try:
        importlib.import_module(lib)
        log_verbosity_handler(lib)
        return True
    except ImportError:
        return False
    except Exception as err:
        print(f"Failure to import {lib}.")
        print(err)


def log_verbosity_handler(library: str, level: int = 50) -> None:
    """Utility to set the logger level of a library to a certain level

    Args:
        library (str): name of the library's logger
        level (int): level to set the logging to. Default to 50 (error level)
    """
    logger = logging.getLogger(library)
    logger.setLevel(level)
