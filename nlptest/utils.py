def is_module_importable(module_name):
    # Return True if module is importable, i.e. installed, otherwise false
    try:
        exec(f'import {module_name}')
        return True
    except:
        return False
