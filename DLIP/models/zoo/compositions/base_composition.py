import pytorch_lightning as pl
import torch.nn as nn
from torch.nn.modules.container import ModuleList


class BaseComposition(pl.LightningModule):
    """ The BaseComposition class. A new composition should extend this class
        to ensure all needed functionalities.
    """

    def __init__(self):
        super(BaseComposition, self).__init__()
        self.composition = ModuleList()

    def set_optimizers(self, optimizer, lrs=None, metric_to_track=None):
        self.optimizer = optimizer
        self.lrs = lrs
        self.metric_to_track = metric_to_track
        if self.metric_to_track  is None:
            self.metric_to_track = "val/loss"

    def configure_optimizers(self):
        if self.lrs is None and self.metric_to_track is None:
            return {"optimizer": self.optimizer}
        if self.lrs is None:
            return {"optimizer": self.optimizer, "monitor": self.metric_to_track}
        if self.metric_to_track is None:
            return {"optimizer": self.optimizer, "lr_scheduler": self.lrs}
        return {"optimizer": self.optimizer,"lr_scheduler": self.lrs,"monitor": self.metric_to_track}

    def get_progress_bar_dict(self):
        # don't show the running loss (very iritating)
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
    
    def append(self, module: nn.Module):
        self.composition.append(module)
        
    def forward(self, x):
        for module in self.composition:
            x = module.forward(x)
        return x
    
    
