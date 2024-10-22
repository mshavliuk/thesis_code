import argparse
import logging

import wandb

import lightning as L
from src.lightning_module import (
    FinetuneModule,
    PretrainingModule,
)
from src.util.config_namespace import (
    MainConfig,
    read_config,
)
from src.util.data_module import MIMICIIIDataModule
from src.util.wandb import get_run_checkpoint


def renew_checkpoint(config: MainConfig, args: argparse.Namespace):
    """
    Loads checkpoint, instantiates model and data module, and saves them back to wandb with same tags and metadata.
    :param config:
    :param args:
    :return:
    """
    api = wandb.Api()
    run = api.run(f"{config.wandb_project}/{args.run_id}")

    checkpoint, artifact = get_run_checkpoint(run, return_artifact=True)
    # checkpoint ='/tmp/renewed_checkpoint.ckpt'
    data: MIMICIIIDataModule = MIMICIIIDataModule.load_from_checkpoint(
        checkpoint,
        stage=config.stage,
        data_fraction=1.0,
        data_config=config.data_config
    )
    if config.stage == 'pretrain':
        model = PretrainingModule.load_from_checkpoint(
            checkpoint,
            strict=True,
            logger=logging.getLogger(__name__),
            features_info=data.get_features_info(),
            module_config=config.module_config,
        )
    elif config.stage == 'finetune':
        model = FinetuneModule.load_from_checkpoint(
            checkpoint,
            strict=True,
            logger=logging.getLogger(__name__),
            features_info=data.get_features_info(),
        )
    else:
        raise ValueError(f"Unknown stage: {config.stage}")
    
    model: L.LightningModule
    
    trainer = L.Trainer(max_epochs=0)
    
    # fuse two modules together to save as single checkpoint
    trainer.fit(model, datamodule=data)
    new_checkpoint = '/tmp/renewed_checkpoint.ckpt'
    trainer.save_checkpoint(new_checkpoint)
    
    new_artifact = wandb.Artifact(
        name=artifact.collection.name,
        type=artifact.type,
        metadata=artifact.metadata,
        description=artifact.description
    )
    new_artifact.add_file(new_checkpoint, name=checkpoint.split('/')[-1])
    
    # with wandb.init(id=run.id, project=run.project) as ctx:
    #     ctx.log_artifact(new_artifact, aliases=artifact.aliases)
    run.log_artifact(new_artifact, aliases=artifact.aliases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID of the model to renew')
    args = parser.parse_args()
    config = read_config(args.config)
    renew_checkpoint(config, args)
