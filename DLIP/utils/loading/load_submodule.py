import importlib
import os
import sys

def load_submodule(module_path: str):
    """ Loads a class from a given module and object name.

    Args:
        module_path (str): Defines the module path. Expected to start after DLIP.

    Returns:
        (list): The class list.
    """
    py_files = list()
    for file in os.listdir(os.path.dirname(__file__)):
        if '__' in file:
            continue
        elif '.py' in file:
            py_files.append(file.replace('.py', ''))

    all_modules = importlib.import_module(f"{module_path}")

    modules = list()
    
    for item in dir(all_modules):
        if isinstance(getattr(all_modules, item) , type):
            modules.append(getattr(all_modules, item))
    
    return modules