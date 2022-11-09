import logging
from DLIP.utils.dlip_globals import(
    TORCH_LRS_MODULE,
    MODELS_MODULE,
    OBJECTIVES_MODULE,
    TORCH_OBJECTIVE_MODULE,
    TORCH_OPTIMIZERS_MODULE
)
from DLIP.utils.loading.dict_to_config import dict_to_config
from DLIP.utils.loading.load_class import load_class
from DLIP.utils.loading.split_parameters import split_parameters


def load_model(model_params: dict, checkpoint_path_str = None):
    model_args = split_parameters(dict_to_config(model_params), ["params"])["params"]
    if "loss_fcn" in model_params:
        loss_fcn = model_params['loss_fcn']
        loss_class_local, score_local = load_class(OBJECTIVES_MODULE, loss_fcn, return_score=True)
        loss_class_torch, score_torch = load_class(TORCH_OBJECTIVE_MODULE, loss_fcn, return_score=True)
        loss_class = loss_class_local if score_local > score_torch else loss_class_torch
        if loss_class is None:
            raise ModuleNotFoundError(f'Cant find class loss function {model_params["loss_fcn"]}.')
        loss_fcn_args = split_parameters(dict_to_config(model_params), ["loss_fcn"])['loss_fcn']
        loss_fcn_args = split_parameters(dict_to_config(loss_fcn_args), ["params"])['params']
        model_args["loss_fcn"] = loss_class(**loss_fcn_args)
    model_type = load_class(MODELS_MODULE, model_params["name"])

    if checkpoint_path_str is None:
        model = model_type(**(model_args))
    else:
        model = model_type.load_from_checkpoint(checkpoint_path=checkpoint_path_str, **(model_args))

    optimizer_config = split_parameters(dict_to_config(model_params), ["optimizer"])
    if "optimizer" in optimizer_config:
        optimizer_config = optimizer_config["optimizer"]
        optimizer = load_class(TORCH_OPTIMIZERS_MODULE, optimizer_config['type'])
        optimizer_params_config = split_parameters(dict_to_config(optimizer_config))
        optimizer_params = optimizer_params_config['params']
        optimizer_params['params'] = model.parameters()
        optim_instance = optimizer(**optimizer_params_config['params'])
        lrs = None
        lrs_instance = None
        if "lrs" in optimizer_params_config:
            lrs_config = optimizer_params_config['lrs']
            lrs = load_class(TORCH_LRS_MODULE, lrs_config['type'])
            lrs_params_config = split_parameters(dict_to_config(lrs_config))
            lrs_instance = lrs(
                optim_instance,
                **lrs_params_config['params']
            )
        if lrs_instance is not None:
            model.set_optimizers(
                optimizer=optim_instance,
                lrs=lrs_instance
            )
        else:
            model.set_optimizers(
                optimizer=optim_instance
            )
    return model
