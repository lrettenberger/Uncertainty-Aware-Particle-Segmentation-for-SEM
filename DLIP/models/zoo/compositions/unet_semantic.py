from typing import List
import torch
import torch.nn as nn
from DLIP.models.zoo.compositions.unet_base import UnetBase
import wandb

class UnetSemantic(UnetBase):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        loss_fcn: nn.Module,
        encoder_type = 'unet',
        encoder_filters: List = [64, 128, 256, 512, 1024],
        decoder_filters: List = [512, 256, 128, 64],
        decoder_type = 'unet',
        dropout: float = 0.0,
        ae_mode = False,
        pretraining_weights = 'imagenet',
        encoder_frozen=False,
        **kwargs,
    ):
        out_channels = num_classes
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
        
        if num_classes==1:
            self.append(nn.Sigmoid())
        else:
            self.append(nn.Softmax())
        self.ae_mode = ae_mode
        
    def training_step(self, batch, batch_idx):
        x, y_true   = batch
        y_true      = y_true.permute(0, 3, 1, 2)
        if self.ae_mode:
            y_true = x
        y_pred      = self.forward(x)
        loss_n_c    = self.loss_fcn(y_pred, y_true)
        loss        = torch.mean(loss_n_c)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        if self.ae_mode:
            y_true = x
        y_pred = self.forward(x)
        loss_n_c    = self.loss_fcn(y_pred, y_true)
        loss        = torch.mean(loss_n_c)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        if batch_idx == 0 and self.current_epoch%20 == 0:
            self.log_imgs(x,y_pred,y_true)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)
        if self.ae_mode:
            y_true = x
        y_pred = self.forward(x)
        loss_n_c    = self.loss_fcn(y_pred, y_true)
        loss        = torch.mean(loss_n_c)
        self.log("test/score", 1-loss, prog_bar=True, on_epoch=True, on_step=False)
        return 1-loss

    def log_imgs(self,x,y,y_true,max_items=16):
        x_wandb = [wandb.Image(x_item.permute(1,2,0).cpu().detach().numpy()) for x_item in x]
        y_wandb = [wandb.Image(y_item.permute(1,2,0).cpu().detach().numpy()) for y_item in y]
        y_true_wandb = [wandb.Image(y_item.permute(1,2,0).cpu().detach().numpy()) for y_item in y_true]
        wandb.log({
            "x": x_wandb[:max_items],
            "y": y_wandb[:max_items],
            "y_true": y_true_wandb[:max_items]
        })