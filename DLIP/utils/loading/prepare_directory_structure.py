import time
import os
import random
from pathlib import Path


from DLIP.utils.experiment_database.experiment_database import ExperimentDatabase
from DLIP.utils.helper_functions.is_int import is_int

def prepare_directory_structure(
    base_path:str,
    experiment_name:str,
    data_module_name:str,
    model_name:str
    ):
    Path(base_path).mkdir(parents=True, exist_ok=True)
    run_path = os.path.join(
            base_path,
            experiment_name,
            data_module_name,
            model_name,
        )
    time.sleep(random.uniform(0,10))
    config_name = '0000'
    if os.path.exists(run_path) and os.path.isdir(run_path):
        dir_content_filtered = [
            x for x in os.listdir(run_path) 
            if (os.path.isdir(os.path.join(run_path,x)) and is_int(x))
        ]
        last_idx = sorted([int(x) for x in dir_content_filtered])[-1]
        config_name = str(last_idx + 1).zfill(4)

    experiment_dir = os.path.join(run_path, config_name)
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    return experiment_dir, config_name