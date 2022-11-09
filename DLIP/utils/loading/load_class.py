import importlib
from DLIP.utils.loading.load_submodule import load_submodule
from difflib import get_close_matches, SequenceMatcher

def load_class(module_path: str, class_name: str, return_score=False):
    """ Loads a class from a given module and object name.

    Args:
        module_path (str): Defines the module path. Expected to start after DLIP.
        class_name (str): The name of the class to be loaded.

    Raises:
        ModuleNotFoundError: If the given class_name is not found

    Returns:
        (Class): The class.
    """
    module_lst = load_submodule(f"{module_path}")
    module_names = [module.__name__ for module in module_lst]
    best_matches = get_close_matches(class_name,module_names)
    if len(best_matches)>0:
        if return_score:
            score = SequenceMatcher(None, class_name, best_matches[0]).ratio()
            return module_lst[module_names.index(best_matches[0])], score
        else:
            return module_lst[module_names.index(best_matches[0])]
    return None, 0