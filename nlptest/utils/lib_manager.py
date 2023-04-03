import importlib


def try_import_lib(lib: str) -> bool:
    """"""
    try:
        importlib.import_module(lib)
        return True
    except ImportError:
        return False
    except Exception as err:
        print(f'Failure to import {lib}.')
        print(err)
