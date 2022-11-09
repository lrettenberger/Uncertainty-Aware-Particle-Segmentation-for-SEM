from wandb import Config


def dict_to_config(parameters_dict: dict):
    parameters_config = Config()
    parameters_config._set_settings("empty")
    for item in dict(parameters_dict).items():
        key, value = item
        parameters_config._items[key] = value
    return parameters_config