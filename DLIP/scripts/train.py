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

# alle labels samples
# checkpoint_path_str='/home/ws/kg2371/projects/sem-segmentation/results/first-shot/GenericSegmentationDataModule/UnetInstance/0059/dnn_weights.ckpt'         

# only good images
# checkpoint_path_str='/home/ws/kg2371/projects/sem-segmentation/results/first-shot/GenericSegmentationDataModule/UnetInstance/0064/dnn_weights.ckpt'         

model = load_model(parameters_splitted["model"],
        #checkpoint_path_str='/home/ws/kg2371/projects/sem-segmentation/results/first-shot/GenericSegmentationDataModule/UnetInstance/0092/dnn_weights.ckpt'         
)

# import torch
# weights = torch.load('/home/ws/kg2371/projects/sem-segmentation/results/first-shot/GenericSegmentationDataModule/UnetSemantic/0077/dnn_weights.ckpt')['state_dict']

# del weights['composition.1.decoder.0.conv.double_conv.0.weight']
# del weights['composition.1.decoder.1.conv.double_conv.0.weight']
# del weights['composition.1.decoder.2.conv.double_conv.0.weight']
# del weights['composition.1.decoder.3.conv.double_conv.0.weight']

# model.load_state_dict(weights,strict=False)

data = load_data_module(parameters_splitted["data"])
trainer = load_trainer(parameters_splitted['train'], experiment_dir, wandb.run.name, data)

# from tqdm import tqdm
# import cv2
# from DLIP.objectives.dice_loss import DiceLoss
# from skimage import measure
# from skimage.segmentation import watershed
# from DLIP.utils.post_processing.distmap2inst import DistMapPostProcessor

# x_true = torch.zeros((0,512,512))
# y_true = torch.zeros((0,512,512))
# y_pred = torch.zeros((0,512,512))
# y_true_inst = torch.zeros((0,512,512))
# y_pred_inst = torch.zeros((0,512,512))

# post_pro = DistMapPostProcessor(**split_parameters(split_parameters(parameters_splitted["model"],['params'])['params'], ["post_pro"])["post_pro"])

# for batch in tqdm(data.test_dataloader()):
#     x,y = batch
#     y_p = model(x)
#     x_true = torch.concat((x_true,x[:,0,:]))
#     y_pred = torch.concat((y_pred,y_p[:,0,:]))
#     y_true = torch.concat((y_true,y[:,:,:,0]))

#     y_p_inst = torch.zeros((0,512,512))
#     y_t_inst = torch.zeros((0,512,512))
#     for i_b in range(len(y)):
#         seeds   = measure.label(y[i_b,:,:,0].cpu().numpy()>0.6, background=0)
#         masks   = y[i_b,:,:,0].cpu().numpy()>0.0
#         gt_mask = watershed(image=-y[i_b,:,:,0].cpu().numpy(), markers=seeds, mask=masks, watershed_line=False)
#         pred_mask = post_pro.process(y_p[i_b,0,:].detach().cpu().numpy(),None)
#         y_p_inst = torch.concat((y_p_inst,torch.Tensor(pred_mask).unsqueeze(0)))
#         y_t_inst = torch.concat((y_t_inst,torch.Tensor(gt_mask).unsqueeze(0)))
#     y_pred_inst = torch.concat((y_pred_inst,y_p_inst))
#     y_true_inst = torch.concat((y_true_inst,y_t_inst))    
# print(f'Accuracy   {torch.sum(y_true == (y_pred>0.6)) / (y_pred.shape[0]*y_pred.shape[1]*y_pred.shape[2]):.2f}')
# print(f'Dice Score {1-DiceLoss()(y_true,y_pred):.2f}')

# for i in tqdm(range(len(x_true))):
#     cv2.imwrite(f'eval/{i}_x.png',(x_true[i]*255).numpy())
#     cv2.imwrite(f'eval/{i}_y_true.png',(y_true[i]*255).numpy())
#     cv2.imwrite(f'eval/{i}_y_pred.png',(y_pred[i]*255).detach().cpu().numpy())
#     cv2.imwrite(f'eval/{i}_y_true_i.png',(y_true_inst[i]*(255/torch.max(y_true_inst[i]))).numpy())
#     cv2.imwrite(f'eval/{i}_y_pred_i.png',(y_pred_inst[i]*(255/torch.max(y_pred_inst[i]))).detach().cpu().numpy())

# exit()

if 'train.cross_validation.n_splits' in cfg_yaml:
    cv_trainer = CVTrainer(
        trainer=trainer,
        n_splits=cfg_yaml['train.cross_validation.n_splits']['value']
    )
    cv_trainer.fit(model=model,datamodule=data)
else:
    trainer.fit(model, data)
    test_results = trainer.test(dataloaders=data.test_dataloader(),ckpt_path='best')
    wandb.log(test_results)
wandb.finish()
