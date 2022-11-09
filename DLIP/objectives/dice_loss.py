import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """ Default Dice Loss. 

    """
    def __init__(self, eps=1e-6) -> None:
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor
        ):
        """
        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        bs      = y_true.size(0)
        cl      = y_true.size(1)
        dims    = (0, 2)

        y_true  = y_true.view(bs, cl, -1) #NxCx(H*W)
        y_pred  = y_pred.view(bs, cl, -1) #NxCx(H*W)

        intersection    = torch.sum(y_pred * y_true, dims)  #C
        cardinality     = torch.sum(y_pred + y_true, dims)  #C

        dice_score = (2. * intersection + self.eps) / (cardinality + self.eps)
        return (1. - dice_score).mean()
