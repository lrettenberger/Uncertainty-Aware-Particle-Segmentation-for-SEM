import yaml
import os

def _extend_dict(extend_me, extend_by):
    for k, v in extend_by.items():
        extend_me[k] = v

def merge_configs(config_arg):
    if isinstance(config_arg, str):
        config_files = config_arg.split()
    configs = []
    for config_file in config_files:
        config_file = os.path.expandvars(config_file)
        with open(config_file) as file:
            configs.append(yaml.load(file, Loader=yaml.FullLoader))
    for i in range(1,len(configs)):
        _extend_dict(configs[0],configs[i])
    return configs[0]
