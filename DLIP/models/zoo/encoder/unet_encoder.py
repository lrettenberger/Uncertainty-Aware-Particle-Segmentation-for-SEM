from typing import List
from torch.nn.modules.container import ModuleList
import torch.nn as nn
import logging
import numpy as np

from DLIP.models.zoo.building_blocks.double_conv import DoubleConv
from DLIP.models.zoo.building_blocks.down_sample import Down

class UnetEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        encoder_filters: List = [64, 128, 256, 512, 1024],
        dropout: float = 0,
        bilinear: bool = False
    ):
        super().__init__()
        if bilinear == True:
            logging.info("Bilinear Upsampling is currently not supported. Ignoring.")
        self.bilinear = False
        factor = 2 if self.bilinear else 1
        encoder_filters = [input_channels] + encoder_filters
        factors = (len(encoder_filters)-2)*[1] + [factor]
        dropout_iter = self.get_dropout_iter(dropout, encoder_filters)
        
        self.backbone = ModuleList()
        self.backbone.append(DoubleConv(
            encoder_filters[0], 
            encoder_filters[1] // factors[0],
            dropout=next(dropout_iter)
        ))
        for i in range(1,len(encoder_filters)-1):
            self.backbone.append(Down(
                encoder_filters[i], 
                encoder_filters[i+1] // factors[i],
                dropout=next(dropout_iter)
            ))
        

    def forward(self, x):
        skip_connections = []
        down_value = x
        for down in self.backbone:
            skip_connections.insert(0, down(down_value))
            down_value = skip_connections[0]
        return skip_connections.pop(0), skip_connections
    
    def get_dropout_iter(self, dropout: int, encoder_filters: List):
        if isinstance(dropout, float) or isinstance(dropout, int): 
            dropout = [dropout for _ in range(len(encoder_filters[1:]))]

        if isinstance(dropout, np.ndarray): 
            dropout = dropout.tolist()

        if len(dropout)!=len(encoder_filters[1:]):
            raise ValueError("Dropout list mismatch to network decoder depth")
        
        return iter(dropout)
