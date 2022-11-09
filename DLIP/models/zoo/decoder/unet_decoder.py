from itertools import zip_longest
from typing import List
import torch.nn as nn
import numpy as np
from torch.nn.modules.container import ModuleList

from DLIP.models.zoo.building_blocks.out_conv import OutConv
from DLIP.models.zoo.building_blocks.up_sample import Up


class UnetDecoder(nn.Module):
    def __init__(
        self,
        n_classes: int,
        encoder_filters: List = [1024, 512, 256, 128, 64],
        decoder_filters: List = [512, 256, 128, 64],
        dropout=0.0,
        billinear_downsampling_used = False,
        ae_mode = False
    ):
        super().__init__()
        self.ae_mode = ae_mode
        #  We need the filters in reversed order
        encoder_filters = encoder_filters[::-1]
        self.n_classes = n_classes
        self.decoder_filters = decoder_filters
        factor = 2 if billinear_downsampling_used else 1
        
        head_filters = encoder_filters[0]
        in_filters = [head_filters] + list(decoder_filters[:-1])
        skip_filters = list(encoder_filters[1:])
        out_filters = decoder_filters
        
        dropout_iter = self.get_dropout_iter(dropout, encoder_filters)

        self.decoder = ModuleList()
        for in_ch, skip_ch, out_ch in zip_longest(in_filters, skip_filters, out_filters):
            skip_ch = 0 if (skip_ch == None or ae_mode) else skip_ch
            self.decoder.append(
                Up(
                    in_ch,
                    out_ch // factor,
                    billinear_downsampling_used,
                    dropout=next(dropout_iter,0),
                    skip_channels=skip_ch,
                )
            )
        self.decoder.append(OutConv(decoder_filters[-1], n_classes))

    def forward(self, x):
        up_value, skip_connections = x
        for i in range(0, len(self.decoder) - 1):
            if self.ae_mode:
                up_value = self.decoder[i](up_value, None)
            else:
                up_value = self.decoder[i](up_value, skip_connections[i] if i < len(skip_connections) else None)
        logits = self.decoder[-1](up_value)
        return logits

    
    def get_dropout_iter(self, dropout: int, decoder_filters: List):
        if isinstance(dropout, float) or isinstance(dropout, int): 
            dropout = [dropout for _ in range(len(decoder_filters[1:]))]

        if isinstance(dropout, np.ndarray): 
            dropout = dropout.tolist()

        if len(dropout)!=len(decoder_filters[1:]):
            raise ValueError("Dropout list mismatch to network decoder depth")
        
        return iter(dropout)
        


        
