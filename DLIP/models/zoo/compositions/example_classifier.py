from typing import List, Tuple
import torch
import torch.nn as nn

from DLIP.models.zoo.compositions.base_composition import BaseComposition
from DLIP.models.zoo.decoder.unet_decoder import UnetDecoder
from DLIP.models.zoo.encoder.example_encoder import ExampleEncoder
from DLIP.models.zoo.encoder.unet_encoder import UnetEncoder


class ExampleClassifier(BaseComposition):
    
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        image_dimensions: Tuple[int, int],
        loss_fcn: nn.Module
    ):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.append(ExampleEncoder(
            input_channels=input_channels,
            num_classes=num_classes,
            image_dimensions=image_dimensions
        ))
 
    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
