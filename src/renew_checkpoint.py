import argparse
import logging
import warnings

import wandb

from src.util.common import (
    create_data_module,
    create_model_module,
)
from src.util.config import (
    MainConfig,
    read_config,
)
from src.util.data_module import MIMICIIIDataModule
from src.util.wandb import (
    create_data_module_artifact,
    create_model_module_artifact,
    get_checkpoint_from_artifacts,
    get_run_checkpoint,
)

def renew_data_module_artifact(config: MainConfig, args: argparse.Namespace):
    """
    Loads checkpoint, instantiates model and data module, and saves them back to wandb with same tags and metadata.
    :param config:
    :param args:
    :return:
    """
    api = wandb.Api()
    artifact = api.artifact(args.target)
    checkpoint = get_checkpoint_from_artifacts(artifact)
    
    kwargs = {
        'stage': config.stage,
        'logger': logging.getLogger(__name__),
        'data_config': config.data_config
    }
    
    data = MIMICIIIDataModule.load_from_checkpoint(checkpoint, **kwargs)
    with wandb.init(
        config=config.model_dump(mode='json') | {'stage': 'data_module'},
        name=config.name,
        job_type='data_module',
    ) as run:
        create_data_module_artifact(config, data)
        run.use_artifact(artifact)
    


def renew_checkpoint(config: MainConfig, args: argparse.Namespace):
    """
    Loads checkpoint, instantiates model and data module, and saves them back to wandb with same tags and metadata.
    :param config:
    :param args:
    :return:
    """
    api = wandb.Api()
    run = api.run(args.from_run)
    data_checkpoint, data_artifact = get_run_checkpoint(
        run, return_artifact=True, type='data_module')
    model_checkpoint, model_artifact = get_run_checkpoint(
        run, return_artifact=True, type=config.stage)
    
    logger = logging.getLogger(__name__)
    
    if data_checkpoint is None:
        warnings.warn("Data module checkpoint not found. Will try to use model checkpoint instead",
                      stacklevel=2)
        data = create_data_module(config, logger, model_checkpoint)
    else:
        data = create_data_module(config, logger, data_checkpoint)
    
    model = create_model_module(config, logger, data, model_checkpoint)
    
    with wandb.init(id=args.from_run, resume="must"):
        create_data_module_artifact(config, data)
        create_model_module_artifact(config, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--target', type=str, required=True, help='Run ID of the model to renew')
    args = parser.parse_args()
    config = read_config(args.config)
    renew_data_module_artifact(config, args)
    # TODO: renew run configs as well
