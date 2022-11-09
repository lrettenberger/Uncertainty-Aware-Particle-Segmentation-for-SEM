from typing import List
from wandb import Config
import numpy as np

PREFIX_DELIMITER = "."


def split_parameters(
    parameters: Config,
    prefixes: List[str] = None,
    numpy_mode=True,
):
    subcategories = {}
    subcategories["other"] = {}
    flexible_mode = False
    if prefixes is not None:
        for prefix in prefixes:
            subcategories[prefix] = {}
    else:
        flexible_mode = True
    for item in dict(parameters).items():
        key, value = item
        if type(value) == list:
            value = np.array(value) if numpy_mode else value
        key_split = [x.lower() for x in (key.split(PREFIX_DELIMITER, 1))]
        if len(key_split) == 2:
            param_type, param_name = key_split
        else:
            param_type, param_name = "other", key_split[0]
        if param_type in subcategories:
            subcategories[param_type][param_name] = value
        else:
            if flexible_mode:
                subcategories[param_type] = {}
                subcategories[param_type][param_name] = value
            else:
                subcategories["other"][key] = value
    return subcategories
