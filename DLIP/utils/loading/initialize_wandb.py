import tempfile
import yaml
import wandb


def initialize_wandb(cfg_yaml:dict, experiment_dir:str, config_name:str):
    with tempfile.NamedTemporaryFile(mode = "w", suffix='.yaml') as tmp:
        yaml.dump(cfg_yaml, tmp)
        wandb.init(
            config=tmp.name,
            allow_val_change=True,
            project=cfg_yaml['wandb.project_name']['value'],
            entity=cfg_yaml['wandb.entity']['value'],
            tags=cfg_yaml['wandb.tags']['value'],
            notes=cfg_yaml['wandb.notes']['value'],
            mode=cfg_yaml['wandb.mode']['value'],
            dir=experiment_dir,
        )
    wandb.run.name = f"{cfg_yaml['experiment.name']['value']}_{cfg_yaml['data.datamodule.name']['value']}_{cfg_yaml['model.name']['value']}_{config_name}"
    wandb.run.save()
    if 'train.cross_validation.active' in cfg_yaml:
        wandb.tensorboard.patch(tensorboardX=False, pytorch=True, root_logdir=experiment_dir)
    else:
        wandb.tensorboard.patch(tensorboardX=False, pytorch=True)
    return wandb.config