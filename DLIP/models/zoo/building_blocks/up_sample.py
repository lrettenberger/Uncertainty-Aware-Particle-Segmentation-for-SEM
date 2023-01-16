import torch
import torch.nn as nn
import torch.nn.functional as F


from DLIP.models.zoo.building_blocks.double_conv import DoubleConv


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear=True,
        dropout=0.0,
        kernel_size=2,
        stride=2,
        skip_channels=0,
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels + skip_channels, out_channels, in_channels // 2,
                dropout=dropout
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride
            )
            self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels, dropout=dropout)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = x1
        if x2 is not None:
            # input is CHW
            diff_y = x2.size()[2] - x1.size()[2]
            diff_x = x2.size()[3] - x1.size()[3]

            x1 = F.pad(
                x1,
                [       
                    torch.div(diff_x, 2, rounding_mode='floor'),
                    torch.div(diff_x - diff_x, 2, rounding_mode='floor'),
                    torch.div(diff_y, 2, rounding_mode='floor'),
                    torch.div(diff_y - diff_y , 2, rounding_mode='floor'),
                ]
            )
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
