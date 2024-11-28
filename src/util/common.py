import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import dpath
import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from src.lightning_module import (
    FinetuneModule,
    PretrainingModule,
)
from src.util.config import MainConfig
from src.util.data_module import (
    MIMICIIIDataModule,
)


def create_data_module(
    config: MainConfig,
    logger: logging.Logger,
    checkpoint: str | None = None
) -> MIMICIIIDataModule:
    kwargs = {
        'stage': config.stage,
        'logger': logger,
    }
    
    if checkpoint is None:
        kwargs |= {'data_config': config.data_config}
        data = MIMICIIIDataModule(**kwargs)
    elif config.stage == 'finetune':
        kwargs |= {'new_config': config.data_config}
        data = MIMICIIIDataModule.load_from_checkpoint(checkpoint, **kwargs)
    else:
        kwargs |= {'data_config': config.data_config}
        data = MIMICIIIDataModule.load_from_checkpoint(checkpoint, **kwargs)
    return data


def create_model_module(
    config: MainConfig,
    logger: logging.Logger,
    data: MIMICIIIDataModule | None = None,
    checkpoint: str | None = None,
) -> PretrainingModule | FinetuneModule:
    if data is None and checkpoint is None:
        raise ValueError("Either data or checkpoint must be provided.")
    
    if config.stage == 'pretrain':
        model_class = PretrainingModule
    elif config.stage == 'finetune':
        model_class = FinetuneModule
    else:
        raise ValueError(f"Invalid stage: {config.stage}")
    
    model_kwargs = {
                       'module_config': config.module_config,
                       'logger': logger,
                   } | ({'features_info': data.get_features_info()} if data is not None else {})
    
    if checkpoint is not None:
        model = model_class.load_from_checkpoint(
            checkpoint_path=checkpoint,
            strict=False,
            **model_kwargs,
        )
    
    else:
        model = model_class(**model_kwargs)
    return model


def create_wandb_logger(
    args: argparse.Namespace,
    config: MainConfig,
    input_artifacts: Sequence[wandb.Artifact] | None = None,
    config_extra: dict | None = None
) -> WandbLogger:
    offline = getattr(args, 'debug', False) or getattr(args, 'dry_run', False)
    kwargs = {
        'name': config.name,
        'offline': offline,
        'dir': os.environ['TEMP_DIR'] + '/wandb',
        'config': dpath.merge(config.model_dump(mode='json'), config_extra),
        'log_model': not offline,
        'checkpoint_name': f"{config.name}_{config.stage}",
        'job_type': config.stage,
        'notes': config.description,
    }
    
    if getattr(args, 'from_run', False):
        kwargs |= {'resume': 'must', 'id': args.from_run}
    
    kwargs |= config.wandb_logger  # any configs would replace the defaults
    wandb_logger = WandbLogger(**kwargs)
    if input_artifacts is not None and not offline:
        for artifact in input_artifacts:
            wandb_logger.experiment.use_artifact(artifact)
    return wandb_logger


def create_trainer(
    config: MainConfig,
    args: argparse.Namespace = argparse.Namespace(),
    wandb_logger: WandbLogger | bool = False
) -> L.Trainer:
    kwargs = {
        'num_sanity_val_steps': 0,
        'enable_model_summary': False,
        'callbacks': get_callbacks(config),
        'logger': wandb_logger,
    }
    
    if sys.stdout.isatty():
        kwargs['enable_progress_bar'] = True
    else:
        kwargs['enable_progress_bar'] = False
    
    # Set accelerator based on CUDA availability
    if torch.cuda.is_available() and not "accelerator" in config.trainer:
        kwargs['accelerator'] = 'gpu'
        
        # Set precision based on GPU support
        if torch.cuda.is_bf16_supported():
            kwargs['precision'] = 'bf16-mixed'
        elif torch.amp.autocast_mode.is_autocast_available('cuda'):
            kwargs['precision'] = '16-mixed'
        else:
            kwargs['precision'] = '32'
    else:
        kwargs['accelerator'] = 'cpu'
        kwargs['precision'] = '32'
    
    kwargs |= config.trainer  # any configs would replace the defaults
    
    print(f"Using {kwargs['accelerator']} accelerator with {kwargs['precision']} precision")
    
    if getattr(args, 'dry_run', False):
        kwargs['fast_dev_run'] = True
    
    return L.Trainer(**kwargs)


def setup():
    torch.set_float32_matmul_precision('medium')
    if torch.__version__.startswith("2.5.0"):
        # CUDNN is automatically preferred over Efficient Attention but much slower
        # see https://github.com/pytorch/pytorch/pull/138522
        torch.backends.cuda.enable_cudnn_sdp(False)
    
    # TODO: handle signals to exit gracefully
    #   SIGTERM - from slurm
    #   SIGINT - from keyboard or wandb UI


def get_callbacks(config: MainConfig) -> list[L.Callback]:
    early_stop = EarlyStopping(**config.early_stop_callback)
    
    checkpoint_kwargs = {
        'save_top_k': 1,
        'save_weights_only': True,
        'verbose': True,
        'dirpath': Path(os.environ['TEMP_DIR'], 'checkpoints', config.stage),
        'filename': f"{config.name}_{{epoch}}_{{val_loss:.4f}}",
    }
    checkpoint_kwargs |= config.checkpoint_callback
    checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)
    
    return [early_stop, checkpoint_callback]
