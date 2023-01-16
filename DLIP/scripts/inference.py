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
from tqdm import tqdm
import cv2
from DLIP.objectives.dice_loss import DiceLoss
from skimage import measure
from skimage.segmentation import watershed
from DLIP.utils.post_processing.distmap2inst import DistMapPostProcessor


determine_false_pos_and_false_neg = False

logging.basicConfig(level=logging.INFO)
logging.info("Initalizing model")

config_files, result_dir = parse_arguments()


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

seed_everything(seed=cfg_yaml['experiment.seed']['value'])
parameters_splitted = split_parameters(config, ["model", "train", "data"])

model = load_model(parameters_splitted["model"], 
                   checkpoint_path_str='/home/ws/kg2371/projects/sem-segmentation/results/first-shot/GenericSegmentationDataModule/UnetInstance/0049/dnn_weights.ckpt')
data = load_data_module(parameters_splitted["data"])


x_true = torch.zeros((0,1024,1024))
y_true = torch.zeros((0,1024,1024))
y_pred = torch.zeros((0,1024,1024))
y_true_inst = torch.zeros((0,1024,1024))
y_pred_inst = torch.zeros((0,1024,1024))

post_pro = DistMapPostProcessor(**split_parameters(split_parameters(parameters_splitted["model"],['params'])['params'], ["post_pro"])["post_pro"])

for batch in tqdm(data.test_dataloader()):
    if len(batch) == 3:
        x,y,_ = batch
    else:
        x,y = batch
    y_b,y = y
    y_p = model(x)
    x_true = torch.concat((x_true,x[:,0,:]))
    y_pred = torch.concat((y_pred,y_p[:,0,:]))
    y_true = torch.concat((y_true,y[:,:,:,0]))

    y_p_inst = torch.zeros((0,1024,1024))
    y_t_inst = torch.zeros((0,1024,1024))
    for i_b in range(len(y)):
        gt_mask = y[i_b,:,:,0].cpu().numpy()
        pred_mask = post_pro.process(y_p[i_b,0,:].detach().cpu().numpy(),None)
        y_p_inst = torch.concat((y_p_inst,torch.Tensor(pred_mask).unsqueeze(0)))
        y_t_inst = torch.concat((y_t_inst,torch.Tensor(gt_mask).unsqueeze(0)))
    y_pred_inst = torch.concat((y_pred_inst,y_p_inst))
    y_true_inst = torch.concat((y_true_inst,y_t_inst))    


if determine_false_pos_and_false_neg:


    border_size = 10

    for i in range(len(y_pred_inst)):
        y_pred_i = y_pred_inst[i]
        y_true_i = y_true_inst[i]
        for k in np.unique(y_pred_i):
                if k==0:
                    continue
                if np.sum(np.logical_and(y_pred_i==k,y_true_i>0).numpy()) == 0:
                    x,y,w,h = cv2.boundingRect(((y_pred_i==k)*1).numpy().astype(np.uint8))
                    y_start = np.max([(y-border_size),0])
                    y_end = y+h+border_size
                    x_start = np.max([x-border_size,0])
                    x_end = x+w+border_size
                    particle = x_true[i][y_start:y_end,x_start:x_end]
                    cv2.imwrite(f'false_positives/{i}_sample_{int(k)}_class.png',(particle*255).numpy())
                    print()
                    
    for i in range(len(y_pred_inst)):
        y_pred_i = y_pred_inst[i]
        y_true_i = y_true_inst[i]
        for k in np.unique(y_true_i):
                if k==0:
                    continue
                if np.sum(np.logical_and(y_true_i==k,y_pred_i>0).numpy()) == 0:
                    x,y,w,h = cv2.boundingRect(((y_true_i==k)*1).numpy().astype(np.uint8))
                    y_start = np.max([(y-border_size),0])
                    y_end = y+h+border_size
                    x_start = np.max([x-border_size,0])
                    x_end = x+w+border_size
                    particle = x_true[i][y_start:y_end,x_start:x_end]
                    cv2.imwrite(f'false_negatives/{i}_sample_{int(k)}_class.png',(particle*255).numpy())
                    print()



# convert to coloeur

r = np.zeros_like(y_true_inst)
g = np.zeros_like(y_true_inst)
b = np.zeros_like(y_true_inst)
for i in tqdm(np.unique(y_true_inst)):
    if i==0:
        continue
    r = np.add(r,((y_true_inst == i)*np.random.randint(0,256)))
    g = np.add(g,((y_true_inst == i)*np.random.randint(0,256)))
    b = np.add(b,((y_true_inst == i)*np.random.randint(0,256)))
y_t_color = np.concatenate((np.expand_dims(r,3),np.expand_dims(g,3),np.expand_dims(b,3)),axis=3).astype(np.uint8)

r = np.zeros_like(y_pred_inst)
g = np.zeros_like(y_pred_inst)
b = np.zeros_like(y_pred_inst)
for i in tqdm(np.unique(y_pred_inst)):
    if i==0:
        continue
    r = np.add(r,((y_pred_inst == i)*np.random.randint(0,256)))
    g = np.add(g,((y_pred_inst == i)*np.random.randint(0,256)))
    b = np.add(b,((y_pred_inst == i)*np.random.randint(0,256)))
y_p_color = np.concatenate((np.expand_dims(r,3),np.expand_dims(g,3),np.expand_dims(b,3)),axis=3).astype(np.uint8)

r = np.zeros_like(y_true)
g = np.zeros_like(y_true)
b = np.zeros_like(y_true)
for i in np.unique(y_true):
    if i==0:
        continue
    r = np.add(r,((y_true == i)*np.random.randint(0,256)))
    g = np.add(g,((y_true == i)*np.random.randint(0,256)))
    b = np.add(b,((y_true == i)*np.random.randint(0,256)))
y_t_color = np.concatenate((np.expand_dims(r,3),np.expand_dims(g,3),np.expand_dims(b,3)),axis=3).astype(np.uint8)

for i in tqdm(range(len(x_true))):
    cv2.imwrite(f'eval/{i}_x.png',(x_true[i]*255).numpy())
    cv2.imwrite(f'eval/{i}_y_true.png',(y_true[i]*255).numpy())
    cv2.imwrite(f'eval/{i}_y_true_color.png',y_t_color[i])
    cv2.imwrite(f'eval/{i}_y_true_i.png',(y_true_inst[i]*(255/torch.max(y_true_inst[i]))).numpy())
    cv2.imwrite(f'eval/{i}_y_pred.png',(y_pred[i]*255).detach().cpu().numpy())
    cv2.imwrite(f'eval/{i}_y_pred_i.png',(y_pred_inst[i]*(255/torch.max(y_pred_inst[i]))).detach().cpu().numpy())
    cv2.imwrite(f'eval/{i}_y_pred_color.png',y_p_color[i])