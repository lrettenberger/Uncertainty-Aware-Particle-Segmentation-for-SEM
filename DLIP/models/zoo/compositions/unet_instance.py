from typing import List
import torch
import torch.nn as nn
from DLIP.models.zoo.compositions.unet_base import UnetBase
from DLIP.utils.post_processing.distmap2inst import DistMapPostProcessor
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.metrics.inst_seg_metrics import get_fast_aji_plus, remap_label
from skimage.segmentation import watershed
from skimage import measure
import wandb

import numpy as np

class UnetInstance(UnetBase):
    def __init__(
        self,
        in_channels: int,
        loss_fcn: nn.Module,
        encoder_type = 'unet',
        encoder_filters: List = [64, 128, 256, 512, 1024],
        decoder_filters: List = [512, 256, 128, 64],
        decoder_type = 'unet',
        dropout: float = 0.0,
        inst_seg_warmup_epochs = 10,
        ae_mode = False,
        pretraining_weights = 'imagenet',
        encoder_frozen=False,
        **kwargs,
    ):
        out_channels = 1
        super().__init__(
                in_channels,
                out_channels,
                loss_fcn,
                encoder_type,
                encoder_filters,
                decoder_filters,
                decoder_type,
                dropout,
                ae_mode,
                pretraining_weights,
                encoder_frozen,
                **kwargs)
        self.append(nn.Sigmoid())
        self.post_pro = DistMapPostProcessor(**split_parameters(kwargs, ["post_pro"])["post_pro"])
        self.inst_seg_warmup_epochs = inst_seg_warmup_epochs
        
    def training_step(self, batch, batch_idx):
        x, y_true   = batch
        y_true      = y_true.permute(0, 3, 1, 2)
        y_pred      = self.forward(x)
        y_true_b = (y_true>0)*1.
        loss_n_c    = self.loss_fcn(y_pred, y_true_b)
        loss        = torch.mean(loss_n_c)
        metric = calc_instance_metric(y_true,y_pred,self.post_pro)
        self.log("train/loss", (loss + (1-metric))/2, prog_bar=True,on_epoch=True)
        return (loss + (1-metric))/2

    def validation_step(self, batch, batch_idx):
        x, y_true   = batch
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = self.forward(x)
        y_true_b = (y_true>0)*1.
        loss_n_c    = self.loss_fcn(y_pred, y_true_b)
        loss        = torch.mean(loss_n_c)
        metric = calc_instance_metric(y_true,y_pred, self.post_pro)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/aji", metric, prog_bar=True, on_epoch=True)
        if batch_idx == 0 and self.current_epoch%10==0:
            self.log_imgs(x,y_pred,y_true)
        return  1-metric

    def test_step(self, batch, batch_idx):
        x, y_true   = batch
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = self.forward(x)
        metric = calc_instance_metric(y_true,y_pred, self.post_pro)
        self.log("test/aji+", metric, prog_bar=True, on_epoch=True)
        return metric
    
    def log_imgs(self,x,y,y_true,max_items=5):
        x_wandb = [wandb.Image(x_item.permute(1,2,0).cpu().detach().numpy()) for x_item in x]
        y_wandb = [wandb.Image(y_item.permute(1,2,0).cpu().detach().numpy()) for y_item in y]
        y_true_wandb = [wandb.Image(y_item.permute(1,2,0).cpu().detach().numpy()) for y_item in y_true]
        wandb.log({
            "x": x_wandb[:max_items],
            "y": y_wandb[:max_items],
            "y_true": y_true_wandb[:max_items]
        })

def calc_instance_metric(y_true,y_pred, post_pro):
    metric = list()
    for i_b in range(y_true.shape[0]):
        gt_mask = y_true[i_b,0,:].cpu().numpy()
        pred_mask = post_pro.process(y_pred[i_b,0,:].detach().cpu().numpy(),None)
        try:
            metric.append(get_fast_aji_plus(remap_label(gt_mask),remap_label(pred_mask)))
        except:
            metric.append(0)
    return np.mean(metric)

def calc_instance_metric_new(y_true,y_pred, post_pro, return_num_elements, weights):
    metric = list()
    num_elements_diff = list()
    for i_b in range(y_true.shape[0]):
        weight = weights[i_b,:,:,0]
        gt_mask = y_true[i_b,0,:].cpu().numpy()
        pred_mask = post_pro.process(y_pred[i_b,0,:].detach().cpu().numpy(),None)
        weight_loss = 1-(np.sum((gt_mask == pred_mask) * weight.detach().cpu().numpy()) / float(torch.sum(weight)))
        if return_num_elements:
            gt_num = len(np.unique(gt_mask))
            pred_num = len(np.unique(pred_mask))
            diff = np.abs(gt_num-pred_num)
            num_elements_diff.append(min([gt_num, diff])/gt_num)
        try:
            metric.append(get_fast_aji_plus(remap_label(gt_mask),remap_label(pred_mask)))
        except:
            metric.append(0)
    if return_num_elements:
        return np.mean(metric),np.mean(num_elements_diff),weight_loss
    return np.mean(metric)


# import cv2
# for i_b in range(y_true.shape[0]):
#     gt_mask = y_true[i_b,0,:].cpu().numpy()
#     pred_mask = post_pro.process(y_pred[i_b,0,:].detach().cpu().numpy(),None)
#     cv2.imwrite(f'results_new/{i_b}_gt.png',gt_mask*(255/np.max(gt_mask)))
#     cv2.imwrite(f'results_new/{i_b}_pred.png',pred_mask*(255/np.max(pred_mask)))