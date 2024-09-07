import argparse
import logging

import lightning as L
import wandb
from lightning.pytorch.loggers import WandbLogger

from src.models.lightning_module import FinetuneStrats
from src.util.config_namespace import read_config
from src.util.data_module import MIMICIIIDataModule


def rerun_tests():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    config = read_config(args.config)
    api = wandb.Api()
    runs = api.runs("Strats", filters={
        "config.name": config.name,
        "summary_metrics.test_epoch_mean_prediction": {"$exists": False},
    })
    for run in runs:
        run: wandb.apis.public.Run
        model_artifacts = [artifact for artifact in run.logged_artifacts() if
                                artifact.type in {'checkpoint', 'model'}]
        assert len(model_artifacts) == 1, f"Expected 1 model artifact, got {len(model_artifacts)}"
        entry = next(iter(model_artifacts[0].manifest.entries))
        checkpoint = model_artifacts[0].get_entry(entry).download()
        wandb_logger = WandbLogger(
            id=run.id,
            project='Strats',
        )
        
        data = MIMICIIIDataModule.load_from_checkpoint(
            checkpoint,
            stage=run.config['stage'],
            data_fraction=run.config['data_fraction'],
            data_config=config.data_config,
        )
        #  FIXME: use PretrainingStrats class if stage is pretrain
        model = FinetuneStrats.load_from_checkpoint(
            checkpoint,
            strict=True,
            logger=logging.getLogger(__name__),
        )
        trainer = L.Trainer(
            **config.trainer,
            logger=[wandb_logger],  # csv_logger
            num_sanity_val_steps=0,
            deterministic=True,
        )
        trainer.test(
            model=model,
            datamodule=data,
        )
        wandb_logger.experiment.finish()


if __name__ == '__main__':
    rerun_tests()
