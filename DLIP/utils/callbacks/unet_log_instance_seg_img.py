from pytorch_lightning import Callback, LightningModule, Trainer
from skimage.color import label2rgb
import wandb
import numpy as np
from DLIP.utils.visualization.inst_seg_contour import visualize_instances_map
from skimage.transform import resize

class UNetLogInstSegImgCallback(Callback):
    """
    Logs one batch of validation dataset.
    """
    def __init__(
        self,
        num_img_log=5
        ):
        """
        Args:
        """
        super().__init__()
        self.num_img_log = num_img_log
        


    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:

        # do prediction
        #pl_module.eval()
        prediction_lst_ol = list()
        prediction_lst_ct = list()
        gt_lst_ol         = list()
        gt_lst_ct         = list()

        for ix in range(min(self.num_img_log,len(trainer.datamodule.test_dataset))):
            x, _ = trainer.datamodule.test_dataset[ix]

            prediction = pl_module(x.unsqueeze(0).to(pl_module.device))

            trainer.datamodule.test_dataset.label_raw_mode = True
            trainer.datamodule.test_dataset.raw_mode = True
            x , gt_mask = trainer.datamodule.test_dataset[ix]
            gt_mask = gt_mask.squeeze()
            pred_mask = pl_module.post_pro.process(prediction[0].squeeze().cpu().numpy(), None)
            trainer.datamodule.test_dataset.label_raw_mode = True
            trainer.datamodule.test_dataset.raw_mode = False

            img_resized = (resize(x, pred_mask.shape)*255).astype(np.uint8)

            y_inst_pred_rgb = label2rgb(pred_mask, bg_label=0, image=img_resized, image_alpha=0.5)
            y_inst_gt_rgb   = label2rgb(gt_mask, bg_label=0, image=img_resized, image_alpha=0.5)

            prediction_lst_ol.append(wandb.Image(y_inst_pred_rgb.copy(), caption=f"{ix+1}") )
            gt_lst_ol.append(wandb.Image(y_inst_gt_rgb.copy(), caption=f"{ix+1}"))
            
            gt_lst_ct.append(
                wandb.Image(
                    visualize_instances_map(img_resized,gt_mask),
                    caption=f"{ix+1}")
            )

            prediction_lst_ct.append(
                wandb.Image(
                    visualize_instances_map(img_resized,pred_mask),
                    caption=f"{ix+1}")
            )

        wandb.log({
            f"test/pred_overlay": prediction_lst_ol, 
            f"test/pred_contour": prediction_lst_ct, 
            f"test/gt_overlay": gt_lst_ol, 
            f"test/gt_contour": gt_lst_ct
        })

        trainer.datamodule.test_dataset.label_raw_mode = False