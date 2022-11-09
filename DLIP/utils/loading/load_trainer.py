from argparse import Namespace
from typing import Any
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer.trainer import Trainer

from DLIP.utils.callbacks.callback_compose import CallbackCompose
from DLIP.utils.loading.dict_to_config import dict_to_config
from DLIP.utils.loading.split_parameters import split_parameters

def load_trainer(train_params: dict, result_dir: str, run_name: str, data:Any = None):

    callback_args = split_parameters(
        dict_to_config(train_params),
        ["callbacks"]
    )["callbacks"]
    callback_args['experiment_dir'] = result_dir
    trainer_args = split_parameters(
        dict_to_config(train_params),["trainer"])["trainer"]
    
    logger = TensorBoardLogger(
        save_dir=result_dir,
        name=run_name,
    )
    
    callbacks = CallbackCompose(Namespace(**callback_args),data)
    instance = Trainer(
        **trainer_args,
        default_root_dir=result_dir,
        logger=logger,
        callbacks=callbacks.get_composition()
    )
    return instance
