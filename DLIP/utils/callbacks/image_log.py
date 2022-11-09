from typing import Sequence
import torch
import torchvision
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor


class ImageLogCallback(Callback):
    """
    Logs one batch of validation dataset.
    """
    def __init__(
        self,
        log_gt_img_once=True,
        ):
        """
        Args:
        """
        super().__init__()
        self.log_gt_img_once = log_gt_img_once
        self.already_logged_gt = False
        


    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # show images of only first batch
        if batch_idx != 0:  
            return

        # get batch
        x, y_true = batch
        y_true = y_true.permute(0, 3, 1, 2)

        # do prediction
        pl_module.eval()
        y_pred = pl_module(x.to(pl_module.device))
        y_pred = y_pred.detach()
        pl_module.train()
        # plot image grids
        self._plot(x,y_true,y_pred,trainer,pl_module.n_classes)

    def _plot(
        self,
        x: Tensor,
        y_true: Tensor,
        y_pred: Tensor,
        trainer: Trainer,
        n_classes: int,
    ) -> None:
        # normalize between zero and 1
        for ib in range(x.shape[0]):
            for ic in range(x.shape[1]):
                min_val = torch.min(x[ib,ic,:])
                max_val = torch.max(x[ib,ic,:])
                if max_val != min_val:
                    x[ib,ic,:] = (x[ib,ic,:]-min_val)/(max_val-min_val)

        img_grid_x = torchvision.utils.make_grid(x)
        if False in (self.log_gt_img_once, self.already_logged_gt):
            trainer.logger.experiment.add_image(
                "x", 
                img_grid_x, 
                global_step=trainer.current_epoch
            )
            
        if n_classes>1:
            img_grid_y_true = torchvision.utils.make_grid(
                y_true.argmax(axis=1).unsqueeze(1)
            )

            img_grid_y_pred = torchvision.utils.make_grid(
                y_pred.argmax(axis=1).unsqueeze(1)
            )
        else:
            img_grid_y_true = torchvision.utils.make_grid(
                y_true
            )

            img_grid_y_pred = torchvision.utils.make_grid(
                torch.round(y_pred)
            )

        if False in (self.log_gt_img_once, self.already_logged_gt):
            trainer.logger.experiment.add_image(
                "y_true", 
                img_grid_y_true, 
                global_step=trainer.current_epoch)
            self.already_logged_gt = True


        trainer.logger.experiment.add_image(
            "y_pred", 
            img_grid_y_pred, 
            global_step=trainer.current_epoch)
