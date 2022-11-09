from typing import Sequence
from pytorch_lightning import Callback, LightningModule, Trainer
import time


class EpochDurationLogCallback(Callback):
    """
    Logs one batch of validation dataset.
    """
    def __init__(
        self
        ):
        """
        Args:
        """
        super().__init__()
        self.start_time = None


    def on_train_epoch_start(        
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        self.start_time = time.time()


    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
    ) -> None:
        duration = time.time() - self.start_time
        pl_module.log("train_duration", duration, prog_bar=True)
