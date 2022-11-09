from pytorch_lightning import Callback, LightningModule, Trainer
import numpy as np


class LogBestMetricsCallback(Callback):
    """
    Logs best metrics
    """
    def __init__(
        self,
        metric_dict
        ):
        """
        Args:
        """
        super().__init__()
        self.metric_dict = dict()
        for metric_name in metric_dict.keys():
            self.metric_dict[metric_name] = dict()
            self.metric_dict[metric_name]["mode"] = metric_dict[metric_name]
            if metric_dict[metric_name]=="max":
                self.metric_dict[metric_name]["best"] = -np.inf
            elif metric_dict[metric_name]=="min":
                self.metric_dict[metric_name]["best"] = np.inf
            else:
                raise ValueError("wrong mode in dict entry")

    def on_validation_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        metrics = trainer.logger_connector.callback_metrics

        for metric_name in self.metric_dict.keys():
            value_epoch = metrics[metric_name].item()
            if self.metric_dict[metric_name]["mode"]=="max":
                if value_epoch>self.metric_dict[metric_name]["best"]:
                    self.metric_dict[metric_name]["best"] = value_epoch
            elif self.metric_dict[metric_name]["mode"]=="min":
                if value_epoch<self.metric_dict[metric_name]["best"]:
                    self.metric_dict[metric_name]["best"] = value_epoch



    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:

        for metric_name in self.metric_dict.keys():
            name = metric_name + "_best"
            trainer.logger.experiment.add_scalar(
                name, 
                self.metric_dict[metric_name]["best"], 
                global_step=trainer.global_step
            )
        
        trainer.final_scores = self.metric_dict
