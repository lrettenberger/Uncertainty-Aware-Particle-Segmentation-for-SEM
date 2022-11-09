import numpy as np
from torchmetrics.functional import dice_score
import torch

def get_dsc(true, pred, eps=1e-6):
    """ DSC Score
    """

    bs      = true.size(0)
    cl      = true.size(1)
    dims    = (0, 2)

    true    = true.view(bs, cl, -1) #NxCx(H*W)
    pred    = pred.view(bs, cl, -1) #NxCx(H*W)

    intersection    = torch.sum(pred * true, dims)  #C
    cardinality     = torch.sum(pred + true, dims)  #C

    dsc_score = (2. * intersection + eps) / (cardinality + eps)

    return dsc_score.mean().item()