from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from DLIP.utils.callbacks.epoch_duration_log import EpochDurationLogCallback
from DLIP.utils.callbacks.image_log import ImageLogCallback
from DLIP.utils.callbacks.log_best_metric import LogBestMetricsCallback

from DLIP.utils.callbacks.unet_log_instance_seg_img import UNetLogInstSegImgCallback

class CallbackCompose:
    def __init__(
        self,
        params,
        data
    ):
        self.params = params
        self.callback_lst = None
        self.data = data
        self.make_composition()


    def make_composition(self):
        self.callback_lst = []

        if hasattr(self.params, 'save_k_top_models'):
            self.weights_dir = self.params.experiment_dir
            self.weights_name = 'dnn_weights'
            self.callback_lst.append(
                ModelCheckpoint(
                    filename = self.weights_name,
                    dirpath = self.weights_dir,
                    save_top_k=self.params.save_k_top_models,
                    monitor='val/loss'
                )
            )

        if hasattr(self.params, 'early_stopping_enabled') and self.params.early_stopping_enabled:
            self.callback_lst.append(
                EarlyStopping(
                monitor='val/loss',
                patience=self.params.early_stopping_patience,
                verbose=True,
                mode='min'
                )
            )

        if hasattr(self.params, 'img_log_enabled')  and self.params.img_log_enabled:
            self.callback_lst.append(
                ImageLogCallback()
            )

        if hasattr(self.params, 'best_metrics_log_enabled') and self.params.best_metrics_log_enabled:
            self.callback_lst.append(
                LogBestMetricsCallback(
                    self.params.log_best_metric_dict
                )
            )

        if hasattr(self.params, 'epoch_duration_enabled') and self.params.epoch_duration_enabled:
            self.callback_lst.append(
                EpochDurationLogCallback()
            )
            
        if hasattr(self.params, 'unet_inst_seg_img_log_enabled') and self.params.unet_inst_seg_img_log_enabled:
            self.callback_lst.append(
                UNetLogInstSegImgCallback()
            )    

    def get_composition(self):
        return self.callback_lst
