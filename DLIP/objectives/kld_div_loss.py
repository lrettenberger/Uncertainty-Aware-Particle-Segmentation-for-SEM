import torch
from torch.functional import Tensor
import torch.nn as nn

class KLDivLossImageData(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.eps=1e-10
        
    def KL(self, P, Q):
        P+=self.eps
        Q+=self.eps
        return torch.sum(P*torch.log(P/Q), dim=[1,2])

    def forward(self, input: Tensor, target: Tensor):
        N,C,H,W = target.shape
        #input = torch.log(input)
        # torch.reshape(torch.permute(input,(0,2,3,1)),(16,512*512,2))
        input_reshaped = input.permute(0,2,3,1).view(N, H*W, C).clone()
        target_reshaped = target.permute(0,2,3,1).view(N, H*W, C).clone()
        return self.KL(input_reshaped, target_reshaped)
        
