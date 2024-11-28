import logging
import shutil
import sys
import warnings
from pathlib import Path
from typing import Literal

import dpath
import lightning as L
import wandb
import yaml
from lightning.fabric.plugins import TorchCheckpointIO

from src.lightning_module import AbstractModule
from src.util.config import MainConfig
from src.util.data_module import MIMICIIIDataModule

DATA_MODULE = 'data_module'
type CheckpointType = Literal[DATA_MODULE, 'model']


def get_run_checkpoint(
    run: wandb.apis.public.Run,
    type: CheckpointType,
    return_artifact=False,
):
    model_artifacts = [artifact for artifact in run.logged_artifacts() if
                       artifact.type == type]
    
    if len(model_artifacts) > 1:
        warnings.warn("Multiple artifacts found. Using the last one", stacklevel=2)
    elif len(model_artifacts) == 0:
        raise FileNotFoundError("Artifact does not contain checkpoint")
    
    artifact = model_artifacts[-1]
    
    checkpoint = get_checkpoint_from_artifacts(artifact)
    
    if return_artifact:
        return checkpoint, artifact
    else:
        return checkpoint


def get_checkpoint_from_artifacts(artifact: wandb.Artifact):
    assert len(artifact.manifest.entries) > 0, "Artifact does not contain any entries"
    
    entry_name = next(iter(artifact.manifest.entries))
    
    checkpoint = artifact.get_entry(entry_name).download()
    return checkpoint


def create_model_artifact(config: MainConfig, module: AbstractModule):
    model_artifact = wandb.Artifact(
        name=f"{config.name}_{config.stage}",
        # has to be 'model' to match wandb logger implementation, see lightning/pytorch/loggers/wandb.py:671
        type="model",
        metadata=config.module_config.model_dump(mode='json'),
    )
    checkpoint = {
        "pytorch-lightning_version": L.__version__,
        'hyper_parameters': module.hparams,
        'state_dict': module.state_dict(),
    }
    io = TorchCheckpointIO()
    with model_artifact.new_file(name='model.ckpt', mode='wb') as f:
        io.save_checkpoint(checkpoint, f.name)
    model_artifact.save()
    model_artifact.wait()
    return checkpoint, model_artifact


def create_data_module_artifact(config: MainConfig, data: MIMICIIIDataModule):
    if data.scalers is None:
        # initialize scalers
        data.setup('fit')
    
    artifact_name, alias = (config.data_module_checkpoint.split(':', 1) + [None])[:2]
    data_artifact = wandb.Artifact(
        name=artifact_name,
        type=DATA_MODULE,
        metadata=config.data_config.model_dump(mode='json'),
    )
    
    checkpoint = {
        "pytorch-lightning_version": L.__version__,
        data.__class__.__qualname__: data.state_dict(),
        data.CHECKPOINT_HYPER_PARAMS_KEY: dict(data.hparams),
    }
    if hparams_name := getattr(data, '_hparams_name'):
        checkpoint[hparams_name] = hparams_name
    
    io = TorchCheckpointIO()
    
    with data_artifact.new_file(name=f'{DATA_MODULE}.ckpt', mode='wb') as f:
        file_path = Path(f.name)
        io.save_checkpoint(checkpoint, file_path)
    
    data_artifact.save()
    data_artifact.wait()
    if alias is not None and alias not in data_artifact.aliases:
        data_artifact.aliases += [alias]
        data_artifact.save()
    
    # move the file to the cached artifact directory so it can be used by next runs without downloading
    artifact_dir = Path(data_artifact._default_root())
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # using shutil to support moving across different filesystems
    shutil.move(str(file_path), str(artifact_dir / file_path.name))
    
    return checkpoint, data_artifact


def create_model_module_artifact(config: MainConfig, model: L.LightningModule):
    model_artifact = wandb.Artifact(
        name=f"{config.name}_{model.stage}",
        # has to be 'model' to match wandb logger implementation, see lightning/pytorch/loggers/wandb.py:671
        type="model",
        metadata=config.module_config.model_dump(mode='json'),
    )
    checkpoint = {
        "pytorch-lightning_version": L.__version__,
        'hyper_parameters': model.hparams,
        'state_dict': model.state_dict(),
    }
    io = TorchCheckpointIO()
    with model_artifact.new_file(name='model.ckpt', mode='wb') as f:
        io.save_checkpoint(checkpoint, f.name)
    model_artifact.save()
    model_artifact.wait()
    return checkpoint, model_artifact


def find_checkpoint(
    config: MainConfig, checkpoint_type: CheckpointType
) -> tuple[str, wandb.Artifact] | tuple[None, None]:
    api = wandb.Api()
    
    try:
        if checkpoint_type == DATA_MODULE:
            artifact = api.artifact(name=config.data_module_checkpoint, type=DATA_MODULE)
        else:
            artifact = api.artifact(name=config.module_checkpoint, type='model')
    except (wandb.errors.CommError, FileNotFoundError) as e:
        return None, None
    
    run = artifact.logged_by()
    checkpoint = get_checkpoint_from_artifacts(artifact)
    
    run: wandb.apis.public.Run | None
    if run is None or run.config is None:
        warnings.warn("Checkpoint run config is missing")
    else:
        compare_configs(config, run.config)
    return checkpoint, artifact


def compare_configs(current: MainConfig, previous: dict):
    current = current.model_dump(mode='json')
    current_diff = {}
    previous_diff = {}
    same_stage = current['stage'] == previous['stage']
    ignore_missing = not same_stage
    
    ignore = {
        'stage',
        'description',
        'data_config/test/loader/batch_size',
        'data_config/val/loader/batch_size',
        # train batch size is not ignored
    }
    
    if not same_stage:
        ignore |= {
            'name',
            'module_config/loss',
            'module_config/model/head_layers/**',
            'module_config/optimizer/**',
            'module_config/model/attention_dropout',
            'data_config/*/loader/**',
            'trainer/**',
            'wandb_logger/**',
            'checkpoint_callback/**',
            'early_stop_callback/**',
        }
    
    ignore = tuple(g.split('/') for g in ignore)
    
    # forward check
    for path, current_value in dpath.segments.leaves(current):
        if any(dpath.segments.match(path, i) for i in ignore):
            continue
        
        if current_value is None and ignore_missing:
            continue
        
        try:
            previous_value = dpath.segments.get(previous, path)
            if previous_value != current_value:
                dpath.new(previous_diff, path, previous_value)
                dpath.new(current_diff, path, current_value)
        except (dpath.exceptions.PathNotFound, KeyError, IndexError):
            if not ignore_missing:
                dpath.new(previous_diff, path, 'Undefined')
                dpath.new(current_diff, path, current_value)
    
    # backward check
    if not ignore_missing:
        for path, value in dpath.segments.leaves(previous):
            if (not dpath.segments.has(current, path) and
                not any(dpath.segments.match(path, i) for i in ignore)):
                dpath.new(current_diff, path, 'Undefined')
                dpath.new(previous_diff, path, value)
    
    if current_diff:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Given configs do not match the artifact's config.\n"
            "Current config difference:\n%s\n"
            "Artifact config difference:\n%s\n",
            yaml.dump(current_diff, sort_keys=True),
            yaml.dump(previous_diff, sort_keys=True),
        )
        
        # if the user is running the script in the terminal, ask for confirmation
        if sys.stdout.isatty():
            print("Do you want to continue? [y/N]: ", end='')
            if input().lower() != 'y':
                raise SystemExit("User aborted the run")
        
        # otherwise look for `grep -r -A 20 "Given configs do not match" .` in the logs directory
