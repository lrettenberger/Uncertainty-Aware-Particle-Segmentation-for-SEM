import os
import wandb
import logging
from pytorch_lightning.utilities.seed import seed_everything

from DLIP.utils.loading.initialize_wandb import initialize_wandb
from DLIP.utils.loading.load_data_module import load_data_module
from DLIP.utils.loading.load_model import load_model
from DLIP.utils.loading.load_trainer import load_trainer
from DLIP.utils.loading.merge_configs import merge_configs
from DLIP.utils.loading.parse_arguments import parse_arguments
from DLIP.utils.loading.prepare_directory_structure import prepare_directory_structure
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.cross_validation.cv_trainer import CVTrainer
from DLIP.utils.post_processing.distmap2inst import DistMapPostProcessor

import numpy as np

logging.basicConfig(level=logging.INFO)
logging.info("Initalizing model")

config_files, result_dir = parse_arguments()

cfg_yaml = merge_configs(config_files)
cfg_yaml['wandb.mode'] = {'value': 'disabled'}
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
        checkpoint_path_str='/home/ws/kg2371/projects/sem-segmentation/results/first-shot/GenericSegmentationDataModule/UnetInstance/0049/dnn_weights.ckpt'         
)
data = load_data_module(parameters_splitted["data"])
trainer = load_trainer(parameters_splitted['train'], experiment_dir, wandb.run.name, data)



post_pro_params = split_parameters(split_parameters(parameters_splitted["model"], ["params"])["params"],['post_pro'])['post_pro']
i = 0
th_seed_start = 0.1
th_seed_end = 0.99
th_seed_step_size = 0.05
th_cell_start = 0.01
th_cell_step_size = 0.002
total_num = sum([len(np.arange(th_cell_start,x,th_cell_step_size)) for x in np.arange(th_seed_start,th_seed_end,th_seed_step_size)])
f = open("hyperparam_results_2012.txt", "a")
# for do_splitting in [True, False]:
for do_splitting in [False]:
    for do_fill_holes in [False]:
    #for do_fill_holes in [True, False]:
        for th_seed in np.arange(th_seed_start,th_seed_end,th_seed_step_size):
            for th_cell in np.arange(th_cell_start,th_seed,th_cell_step_size):
                post_pro_params['th_seed'] = th_seed
                post_pro_params['th_cell'] = th_cell
                post_pro_params['do_splitting'] = do_splitting
                post_pro_params['do_fill_holes'] = do_fill_holes
                model.post_pro = DistMapPostProcessor(**post_pro_params)
                val_results = trainer.test(model,dataloaders=data.val_dataloader(),verbose=False)
                i+=1
                print(f'Progress: {(i/total_num)*100:.2f}%')
                f.write(f'{do_splitting},{do_fill_holes},{th_seed},{th_cell},{val_results[0]["test/aji+"]}\n')
                f.flush()
f.close()