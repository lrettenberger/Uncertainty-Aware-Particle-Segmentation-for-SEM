from typing import List
import torch.nn as nn
import torch
import numpy as np
from torch.nn.modules.container import ModuleList

class BasicEncoder(nn.Module):
    def __init__(self, input_channels:int,classification_output:bool):
        super().__init__()
        self.backbone = ModuleList()
        self.input_channels = input_channels
        self.classification_output = classification_output

    def get_dropout_iter(self, dropout: int, encoder_filters: List):
        if isinstance(dropout, float) or isinstance(dropout, int): 
            dropout = [dropout for _ in range(len(encoder_filters[1:]))]

        if isinstance(dropout, np.ndarray): 
            dropout = dropout.tolist()

        if len(dropout)!=len(encoder_filters[1:]):
            raise ValueError("Dropout list mismatch to network decoder depth")
        
        return iter(dropout)

    def get_skip_channels(self):
        skip_channels = []
        dummy_value = torch.randn((1,self.input_channels,256,256))
        for down in self.backbone:
            dummy_value = down(dummy_value)
            skip_channels.append(dummy_value.shape[1])
        return skip_channels


    def forward(self, x):
        skip_connections = []
        down_value = x
        for down in self.backbone:
            skip_connections.insert(0, down(down_value))
            down_value = skip_connections[0]
        if self.classification_output:
            return skip_connections.pop(0)
        return skip_connections.pop(0), skip_connections  
