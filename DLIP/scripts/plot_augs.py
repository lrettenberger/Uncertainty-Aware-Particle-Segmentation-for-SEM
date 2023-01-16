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
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from DLIP.objectives.dice_loss import DiceLoss
from skimage import measure
from skimage.segmentation import watershed
from DLIP.utils.post_processing.distmap2inst import DistMapPostProcessor


determine_false_pos_and_false_neg = True

logging.basicConfig(level=logging.INFO)
logging.info("Initalizing model")

#config_files, result_dir = parse_arguments()
result_dir = './'
config_files = '/home/ws/kg2371/projects/sem-segmentation/DLIP/experiments/configurations/inst_seg_20.yaml'

cfg_yaml = merge_configs(config_files)
base_path=os.path.expandvars(result_dir)
experiment_name=cfg_yaml['experiment.name']['value']

cfg_yaml['wandb.mode'] = {'value': 'disabled'}

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

parameters_splitted = split_parameters(config, ["model", "train", "data"])



params = {
    'img_processing.aug1.aug_shift_scale_rotate_scale_lim': 0.6,
    'img_processing.aug1.aug_shift_scale_rotate_shift_lim': 0.1,
    'img_processing.aug1.gaussian_blur_sigma_limit': 15,
    'img_processing.aug1.gray_image_alpha': 0.8,
    'img_processing.aug1.aug_rand_brightness_contrast_brightness_limit': 0.5,
    'img_processing.aug1.aug_rand_brightness_contrast_contrast_limit': 1.0,
    'img_processing.aug1.aug_shift_scale_rotate_rot_lim': 45,
    'img_processing.aug1.aug_shift_scale_rotate_scale_lim': 0.6,
    'img_processing.aug1.aug_shift_scale_rotate_shift_lim': 0.1
}

for param in tqdm(params.keys()):
    #for zeroed in params.keys():
    #    parameters_splitted['data'][zeroed] = 0.0
    fig = plt.figure(figsize=(8, 8))
    columns = 8
    rows = 8
    counter = 0
    aug_vals = np.arange(0.0,params[param],params[param]/(columns*rows))
    for i in range(1, columns*rows +1):
        #parameters_splitted['data'][param] = tuple([aug_vals[counter],aug_vals[counter]])
        #parameters_splitted['data'][param] = aug_vals[counter]
        data = load_data_module(parameters_splitted["data"])
        fig.add_subplot(rows, columns, i)
        plt.imshow(data.train_dataloader().dataset[0][0][0],cmap='gray',vmin=0,vmax=1)
        plt.title(f'{aug_vals[counter]:.3f}')
        counter+=1
    plt.savefig(f'{param}.png')
    plt.close()

