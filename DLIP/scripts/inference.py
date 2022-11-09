import os
import wandb
import logging
from pytorch_lightning.utilities.seed import seed_everything
from DLIP.objectives.dice_loss import DiceLoss
import numpy as np
from DLIP.utils.loading.initialize_wandb import initialize_wandb
from DLIP.utils.loading.load_data_module import load_data_module
from DLIP.utils.loading.load_model import load_model
from DLIP.utils.loading.load_trainer import load_trainer
from DLIP.utils.loading.merge_configs import merge_configs
from DLIP.utils.loading.parse_arguments import parse_arguments
from DLIP.utils.loading.prepare_directory_structure import prepare_directory_structure
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.cross_validation.cv_trainer import CVTrainer
import torch


logging.basicConfig(level=logging.INFO)
logging.info("Initalizing model")

config_files, result_dir = parse_arguments()

cfg_yaml = merge_configs(config_files)
base_path=os.path.expandvars(result_dir)
experiment_name=cfg_yaml['experiment.name']['value']

experiment_dir, config_name = prepare_directory_structure(
    base_path=base_path,
    experiment_name=experiment_name,
    data_module_name=cfg_yaml['data.datamodule.name']['value'],
    model_name=cfg_yaml['model.name']['value']
)

config = initialize_wandb(
    cfg_yaml=cfg_yaml,
    experiment_dir=experiment_dir,
    config_name=config_name
)

seed_everything(seed=cfg_yaml['experiment.seed']['value'])
parameters_splitted = split_parameters(config, ["model", "train", "data"])

model = load_model(parameters_splitted["model"], 
                   checkpoint_path_str='/home/ws/nd6488/soft-labeling/sweep/soft-labeling-blur-5 (Hard Labels)/SoftLabelDataModule/UnetSupervised/0006/dnn_weights.ckpt')
data = load_data_module(parameters_splitted["data"])



score = None

dice = DiceLoss()

for batch in data.test_dataloader():
    x,y = batch
    y = y = torch.cat((y,1 - y),3)
    y = y.permute(0,3,1,2)
    y_pred = model(x)
    score = 1 - dice(y_pred,y)
    break
print(score)