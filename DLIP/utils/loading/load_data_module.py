import logging

from DLIP.utils.loading.append_to_transforms import append_to_transforms
from DLIP.utils.loading.dict_to_config import dict_to_config
from DLIP.utils.loading.load_class import load_class
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.dlip_globals import DATA_MODULE


def load_data_module(data_params: dict,  do_val_init=True):
    datamodule_params = split_parameters(
                                dict_to_config(data_params),
                                ["datamodule"])["datamodule"]
    datamodule_args = split_parameters(
                                dict_to_config(datamodule_params),
                                ["arguments"])["arguments"]
    img_processing_args = split_parameters(
                                dict_to_config(data_params),
                                ["img_processing"])["img_processing"]

    transformations = split_parameters(dict_to_config(img_processing_args))
    general_configuration = transformations['other']
    del transformations['other']

    # extract additional validation time transforms
    val_augs = None
    if 'validation_aug' in transformations:
        val_augs = transformations['validation_aug']
        del transformations['validation_aug']

    train_transforms = []
    val_transforms = []
    test_transforms = []
    if len(transformations.values()) > 0:
        for img_processing in transformations.values():
            append_to_transforms(
                                dict(general_configuration, **img_processing),
                                train_transforms,
                                val_transforms,
                                test_transforms,
                                val_augs)
    else:
        append_to_transforms(
                            general_configuration,
                            train_transforms,
                            val_transforms,
                            test_transforms,
                            val_augs)

    datamodule = load_class(DATA_MODULE, datamodule_params["name"])
    # Create new dict with train_transforms and val_transforms
    # and merge it with the datamodule args.
    if 'root_dirs' in data_params and 'device' in datamodule_params:
        datamodule_args['root_dir'] = datamodule_params["root_dirs"][datamodule_params["device"]]
    datamodule = datamodule(
        **{
            **{
                **{
                    "train_transforms": train_transforms,
                    "val_transforms": val_transforms,
                    "test_transforms": test_transforms,
                },
                **datamodule_args,
            }
        }
    )

    if hasattr(datamodule, 'init_val_dataset') and do_val_init:
        datamodule.init_val_dataset()

    if do_val_init:
        logging.info(f"Length of validation dataset: {len(datamodule.val_dataset)}")
        logging.info(f"Length of test dataset: {len(datamodule.test_dataset)}")
    
    return datamodule
