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
        
        self.max = 5
        self.min = 1
        
        
    def scale_range (self,input):
        input += -(torch.min(input))
        input /= torch.max(input) / (self.max - self.min)
        input += self.min
        return input
        
    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            # we have weight maps if len is 3
            x, y_true, weight_maps   = batch
            y_true_b,y_true = y_true
            y_true      = y_true.permute(0, 3, 1, 2)
            y_true_b      = y_true_b.permute(0, 3, 1, 2)
            y_pred      = self.forward(x)
            loss_n_c    = self.loss_fcn(y_pred, y_true_b)
            loss = self.scale_range(weight_maps.unsqueeze(1)) * loss_n_c
            loss = torch.sum(loss) / torch.numel(loss)
            self.log("train/loss", loss, prog_bar=True,on_epoch=True)
            return loss
        else:
            x, y_true   = batch
            y_true_b,y_true = y_true
            y_true      = y_true.permute(0, 3, 1, 2)
            y_true_b      = y_true_b.permute(0, 3, 1, 2)
            y_pred      = self.forward(x)
            loss_n_c    = self.loss_fcn(y_pred, y_true_b)
            loss        = torch.mean(loss_n_c)
            self.log("train/loss", loss, prog_bar=True,on_epoch=True)
            return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            # we have weight maps if len is 3
            x, y_true, weight_maps   = batch
            y_true_b,y_true = y_true
            y_true      = y_true.permute(0, 3, 1, 2)
            y_true_b      = y_true_b.permute(0, 3, 1, 2)
            y_pred      = self.forward(x)
            loss_n_c    = self.loss_fcn(y_pred, y_true_b)
            loss = weight_maps.unsqueeze(1) * loss_n_c
            loss = torch.sum(loss) / torch.numel(loss)
            metric,gt_masks,pred_masks = calc_instance_metric(y_true,y_pred, self.post_pro)
            self.log("val/loss", loss, prog_bar=True, on_epoch=True)
            self.log("val/aji", metric, prog_bar=True, on_epoch=True)
            if batch_idx == 1:
                self.log_imgs(x,y_pred,y_true_b,gt_masks,pred_masks)
            #return  1-metric
            return loss
        else:
            x, y_true   = batch
            y_true_b,y_true = y_true
            y_true = y_true.permute(0, 3, 1, 2)
            y_true_b      = y_true_b.permute(0, 3, 1, 2)
            y_pred = self.forward(x)
            loss_n_c    = self.loss_fcn(y_pred, y_true_b)
            metric,gt_masks,pred_masks = calc_instance_metric(y_true,y_pred, self.post_pro)
            self.log("val/loss", loss, prog_bar=True, on_epoch=True)
            self.log("val/aji", metric, prog_bar=True, on_epoch=True)
            if batch_idx == 1:
                self.log_imgs(x,y_pred,y_true_b,gt_masks,pred_masks)
            #return  1-metric
            return loss

    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y_true, weight_maps   = batch
            y_true_b,y_true = y_true
            y_true = y_true.permute(0, 3, 1, 2)
            y_true_b      = y_true_b.permute(0, 3, 1, 2)
            y_pred = self.forward(x)
            metric,gt_masks,pred_masks = calc_instance_metric(y_true,y_pred, self.post_pro)
            self.log("test/aji+", metric, prog_bar=True, on_epoch=True)
            return metric
        else:
            x, y_true   = batch
            y_true_b,y_true = y_true
            y_true = y_true.permute(0, 3, 1, 2)
            y_true_b      = y_true_b.permute(0, 3, 1, 2)
            y_pred = self.forward(x)
            metric,gt_masks,pred_masks = calc_instance_metric(y_true,y_pred, self.post_pro)
            self.log("test/aji+", metric, prog_bar=True, on_epoch=True)
            return metric
    
    def log_imgs(self,x,y,y_true,gt_masks,pred_masks,max_items=5):
        x_wandb = [wandb.Image(x_item.permute(1,2,0).cpu().detach().numpy()) for x_item in x]
        y_wandb = [wandb.Image(y_item.permute(1,2,0).cpu().detach().numpy()) for y_item in y]
        y_true_wandb = [wandb.Image(y_item.permute(1,2,0).cpu().detach().numpy()) for y_item in y_true]
        for i in range(len(x)):
            mask = wandb.Image(x[i].permute(1,2,0).cpu().detach().numpy(), masks={
                "predictions": {
                    "mask_data": pred_masks[i],
                },
                "ground_truth": {
                    "mask_data": gt_masks[i],
                },
                })
            wandb.log({f'mask_{i}':mask})
        wandb.log({
            "x": x_wandb[:max_items],
            "y": y_wandb[:max_items],
            "y_true": y_true_wandb[:max_items],
        })

def calc_instance_metric(y_true,y_pred, post_pro, return_masks = True):
    metric = list()
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    gt_masks = np.empty_like(y_true[:,0,:])
    pred_masks = np.empty_like(y_pred[:,0,:])
    for i_b in range(y_true.shape[0]):
        gt_mask = y_true[i_b,0,:]
        pred_mask = post_pro.process(y_pred[i_b,0,:],None)
        try:
            metric.append(get_fast_aji_plus(remap_label(gt_mask),remap_label(pred_mask)))
            gt_masks[i_b] = gt_mask
            pred_masks[i_b] = pred_mask
        except:
            metric.append(0)
    return np.mean(metric),gt_masks,pred_masks


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