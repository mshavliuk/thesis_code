import argparse
import logging

import lightning as L
import wandb
from lightning.pytorch.loggers import WandbLogger

from src.lightning_module import (
    FinetuneModule,
    PretrainingModule,
)
from src.util.callbacks import get_callbacks

from src.util.config_namespace import MainConfig
from src.util.data_module import MIMICIIIDataModule
from src.util.wandb import (
    get_checkpoint_from_artifacts,
    get_run_checkpoint,
)


def find_checkpoint(config: MainConfig, args: argparse.Namespace, return_artifact: bool = False):
    api = wandb.Api()
    project = config.wandb_logger['project']
    if hasattr(args, 'from_run') and args.from_run:
        run = api.run(f"{project}/{args.from_run}")
        return get_run_checkpoint(run)
    elif hasattr(config, 'checkpoint') and (artifact_name := config.checkpoint):
        artifact = api.artifact(f"{project}/{artifact_name}", type='model')
        checkpoint = get_checkpoint_from_artifacts([artifact])
        if return_artifact:
            return checkpoint, artifact
        else:
            return checkpoint
    else:
        if return_artifact:
            return None, None
        else:
            return None


def create_data_module(config: MainConfig, logger: logging.Logger, checkpoint: str | None = None) -> MIMICIIIDataModule:
    if checkpoint is not None:
        data = MIMICIIIDataModule.load_from_checkpoint(
            checkpoint,
            stage=config.stage,
            data_config=config.data_config,
            logger=logger,
        )
    else:
        data = MIMICIIIDataModule(
            stage=config.stage,
            data_config=config.data_config,
            logger=logger,
        )
    return data

def create_model_module(
    config: MainConfig,
    logger: logging.Logger,
    data: MIMICIIIDataModule,
    checkpoint: str | None = None,
) -> PretrainingModule | FinetuneModule:
    if config.stage == 'pretrain':
        model_class = PretrainingModule
    elif config.stage == 'finetune':
        model_class = FinetuneModule
    else:
        raise ValueError(f"Invalid stage: {config.stage}")
    
    if checkpoint is not None:
        model = model_class.load_from_checkpoint(
            checkpoint,
            strict=False,
            module_config=config.module_config,
            features_info=data.get_features_info(),
            logger=logger,
        )
    else:
        model = model_class(
            module_config=config.module_config,
            features_info=data.get_features_info(),
            logger=logger,
        )
    return model


def create_wandb_logger(args, config, artifact: wandb.Artifact | None = None):
    offline = args.debug
    if hasattr(args, 'from_run') and args.from_run:
        kwargs = {'resume': 'must', 'id': args.from_run}
    else:
        kwargs = {}
    
    wandb_logger =  WandbLogger(
        **config.wandb_logger,
        name=config.name,
        offline=offline,
        # tags=[args.stage],
        config=config,
        log_model=not offline,
        checkpoint_name=config.name,
        **kwargs,
    )
    if artifact and not offline:
        wandb_logger.experiment.use_artifact(artifact)
    return wandb_logger


def create_trainer(args, config, wandb_logger: WandbLogger):
    return L.Trainer(
        **config.trainer,
        accelerator='cpu' if args.debug else 'gpu',
        num_sanity_val_steps=0,
        enable_model_summary=False,
        callbacks=get_callbacks(config),
        logger=wandb_logger,
    )
