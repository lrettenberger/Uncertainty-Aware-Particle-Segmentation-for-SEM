import logging
import gc
import os
from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection, LightningLoggerBase
from sklearn.model_selection import KFold
import wandb
import numpy as np
import torch

class CVTrainer:
    def __init__(self, trainer: Trainer, n_splits: int):
        super().__init__()
        self._trainer = trainer
        self._n_splits = n_splits
        self._kf = KFold(n_splits=self._n_splits, shuffle=True)
        self.scores = {"val/loss_best": list(), "test/loss": list()}

    @staticmethod
    def _update_logger(logger: LightningLoggerBase, fold_idx: int):
        """
            Change a model logger parameters to log new fold
        Args:
            logger: Logger to update
            fold_idx: Fold ID
        """
        if hasattr(logger, 'experiment_name'):
            logger_key = 'experiment_name'
        elif hasattr(logger, 'name'):
            logger_key = 'name'
        else:
            raise AttributeError('The logger associated with the trainer '
                                 'should have an `experiment_name` or `name` '
                                 'attribute.')
        new_experiment_name = getattr(logger, logger_key) + f'/fold_{fold_idx}'
        logger._name = new_experiment_name

    @staticmethod
    def update_modelcheckpoint(model_ckpt_callback: ModelCheckpoint, fold_idx: int):
        """
            Update model checkpoint object with fold information
        Args:
            model_ckpt_callback: Model checkpoint object
            fold_idx: Fold ID
        """
        _default_filename = '{epoch}-{step}'
        _suffix = f'_fold{fold_idx}'
        if model_ckpt_callback.filename is None:
            new_filename = _default_filename + _suffix
        else:
            new_filename = model_ckpt_callback.filename + _suffix
        setattr(model_ckpt_callback, 'filename', new_filename)

    def update_loggers(self, trainer: Trainer, fold_idx: int):
        """
            Change model's loggers parameters to log new fold
        Args:
            trainer: Trainer whose logger to update
            fold_idx: Fold ID
        """
        if not isinstance(trainer.logger, LoggerCollection):
            _loggers = [trainer.logger]
        else:
            _loggers = trainer.logger

        # Update loggers:
        for _logger in _loggers:
            self._update_logger(_logger, fold_idx)

    def fit(self, model: pl.LightningModule, datamodule: LightningDataModule):
        datamodule.reset_val_dataset()

        for fold_idx, (train_index, val_index) in enumerate(
            self._kf.split(datamodule.labeled_train_dataset)
            ):
            datamodule = deepcopy(datamodule)
            datamodule.init_val_dataset(val_index)

            # Clone model & trainer:
            _model = deepcopy(model)
            _trainer = deepcopy(self._trainer)

            # Update loggers and callbacks:
            #self.update_loggers(_trainer, fold_idx)
            for callback in _trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    self.update_modelcheckpoint(callback, fold_idx)

            _trainer.fit(_model, datamodule)
            self.scores["val/loss_best"].append(_trainer.final_scores["val/loss"]["best"])
            self.scores['test/loss'].append(_trainer.test(_model, datamodule.test_dataloader())[0]['test/loss'])


            del _model, _trainer
            torch.cuda.empty_cache()
            gc.collect()
            
            datamodule.reset_val_dataset()

        self.finalize_results(model, datamodule.test_dataloader())

    def finalize_results(self, model, test_loader):
        logging.info("Finalize results of k-fold cross validation...")
        for callback in self._trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                dirpath = callback.dirpath
                filename = callback.filename

        best_model_fold = np.argmin(self.scores["val/loss_best"])

        best_model_name = f"{filename}_fold{best_model_fold}.ckpt"

        logging.info(f"Selection of best checkpoint obtained in fold idx {best_model_fold}")
        for file in os.listdir(dirpath):
            if file.endswith(".ckpt"):
                file_path = os.path.join(dirpath,file)
                if file != best_model_name:
                    os.remove(file_path)

        file_path =  os.path.join(dirpath,best_model_name)
        os.rename(file_path, file_path.replace(f"_fold{best_model_fold}", ""))

        self._trainer.test(model=model, ckpt_path=file_path, test_dataloaders=test_loader)

        wandb.log({"scores/mean_val_loss": np.mean(self.scores["val/loss_best"])})
        wandb.log({"scores/std_val_loss": np.std(self.scores["val/loss_best"])})
        wandb.log({"scores/mean_test_loss": np.mean(self.scores["test/loss"])})
        wandb.log({"scores/std_test_loss": np.std(self.scores["test/loss"])})
