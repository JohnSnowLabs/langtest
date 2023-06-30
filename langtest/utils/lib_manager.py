import importlib


def try_import_lib(lib: str) -> bool:
    """
    Tries to import a Python library/module dynamically and returns True if successful, False otherwise.

    Args:
        lib (str): The name of the library/module to import.

    Returns:
        bool: True if the import is successful, False otherwise.
    """
    try:
        importlib.import_module(lib)
        return True
    except ImportError:
        return False
    except Exception as err:
        print(f'Failure to import {lib}.')
        print(err)
