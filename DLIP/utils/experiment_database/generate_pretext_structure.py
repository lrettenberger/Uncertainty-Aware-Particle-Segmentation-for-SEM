import os
import yaml

from DLIP.utils.experiment_database.experiment_database import ExperimentDatabase
from DLIP.utils.experiment_database.split_dir_on_lsdf import split_dir_if_on_lsdf


def generate_pretext_structure(
    experiment_name: str,
    name: str,
    experiment_dir: str,
    checkpoint_path: str,
    dataset_name: str,
    ssl_method_name: str,
    config: dict,
    database: ExperimentDatabase
):
    with open(os.path.join(experiment_dir,'config.yaml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)
    return database.insert_pretext(
        experiment_name,
        name,
        dataset_name,
        ssl_method_name,
        split_dir_if_on_lsdf(experiment_dir),
        split_dir_if_on_lsdf(checkpoint_path),
        config
    )