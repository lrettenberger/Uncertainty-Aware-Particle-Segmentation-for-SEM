from typing import Tuple
import torch
import torch.nn as nn

from DLIP.models.zoo.building_blocks.double_conv import DoubleConv

class ExampleEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        image_dimensions: Tuple[int, int]
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.encoder = DoubleConv(input_channels, 32)
        self.fc = nn.Linear(int(32*image_dimensions[0]*image_dimensions[1]), 1)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)
