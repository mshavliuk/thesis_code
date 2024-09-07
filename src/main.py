import argparse
import io
import logging
import os
from typing import Generator
import finetuning_scheduler as fts
import lightning as L
import torch
import wandb
import yaml
from lightning.pytorch import (
    seed_everything,
)
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import (
    CSVLogger,
    WandbLogger,
)

from src.models.lightning_module import (
    FinetuneModuleConfig,
    FinetuneStrats,
    ModuleConfig,
    PretrainingStrats,
)
from src.util.config_namespace import (
    MainConfig,
    read_config,
)
from src.util.data_module import MIMICIIIDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--watch', action='store_true', help='Watch model with wandb')
    parser.add_argument('--folds_number', type=int, default=1, help='Number of folds')
    parser.add_argument('--data_fraction',
                        type=lambda line: tuple(float(v) for v in (line.split(','))),
                        default='1.0',
                        help='Fraction of data to use')
    args = parser.parse_args()
    config = read_config(args.config)
    
    logging.basicConfig(level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    torch.set_float32_matmul_precision('medium')
    if config.stage == 'pretrain':
        model = pretrain(config, logger, args)
    elif config.stage == 'finetune':
        model = finetune(config, logger, args)


def pretrain(hparams: MainConfig, logger: logging.Logger, args: argparse.Namespace):
    seed = seed_everything(42)
    run_name = hparams.name
    csv_logger = CSVLogger(hparams.results_dir, name='pretrain', version=run_name)
    
    if checkpoint_path := hparams.get('checkpoint'):
        model = PretrainingStrats.load_from_checkpoint(
            checkpoint_path, logger=logger, strict=True)
        data = MIMICIIIDataModule.load_from_checkpoint(checkpoint_path)
    else:
        data = MIMICIIIDataModule(
            stage='pretrain',
            data_config=hparams.data_config,
        )
        module_config = ModuleConfig(hparams.module_config)
        model = PretrainingStrats(module_config=module_config,
                                  features_info=data.get_features_info())
    
    early_stop = EarlyStopping(
        **hparams.early_stop_callback,
        verbose=True
    )
    
    wandb_logger = WandbLogger(
        offline=args.debug,
        **hparams.wandb_logger,
        config=hparams,
        log_model=not args.debug,
        checkpoint_name=f"{run_name}_pretrain",
    )
    
    checkpoint_callback = ModelCheckpoint(
        **hparams.checkpoint_callback,
        filename='strats-{epoch}-{val_epoch_loss:.2f}',
        verbose=True,
    )
    if args.watch:
        wandb_logger.watch(model, log='all')
    
    trainer = L.Trainer(
        **hparams.trainer,
        logger=[wandb_logger, csv_logger],
        deterministic=True,
        callbacks=[early_stop, checkpoint_callback],
    )
    if not args.debug and not args.watch:
        model.compile(fullgraph=True)
    
    trainer.fit(
        model=model,
        datamodule=data,
    )
    trainer.test(
        ckpt_path="best",
        datamodule=data,
    )
    
    wandb_logger.experiment.finish()
    
    return model


def finetune(config: MainConfig, logger: logging.Logger, args: argparse.Namespace):
    run_name = config.name
    csv_logger = CSVLogger(
        config.results_dir,
        name='finetune',
        version=run_name
    )
    
    for fold, data_fraction, model, data, wandb_logger in get_fold_states(args, config, logger):
        logger.info(f"Training fold {fold}...")
        # TODO: try to freeze the pretrained backbone and train the head only
        # TODO: try to swap entire forecast head with binary
        
        try:
            fold_run_name = f"{run_name}_frac_{data_fraction}_fold_{fold}"
            finetune_fold(fold_run_name, config, model, data, csv_logger, wandb_logger)
        except Exception as e:
            logger.exception(f"Error in fold {fold}: {e}")


def copy_state(model: torch.nn.Module) -> dict:
    # return {k: v.cpu() for k, v in model.state_dict().items()}
    state_copy = {}
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor):
            if v.is_cuda:
                state_copy[k] = v.cpu()
            else:
                state_copy[k] = v.clone()
        elif isinstance(v, io.BytesIO):
            v.seek(0)
            state_copy[k] = torch.load(v)
        else:
            raise ValueError(f"Unknown state_dict value type: {type(v)}")
    return state_copy

def get_fold_states(
    args: argparse.Namespace, config: MainConfig, logger
) -> Generator[tuple[int, int, FinetuneStrats, MIMICIIIDataModule, WandbLogger], None, None]:
    # TODO: allow pristine model
    if os.path.isfile(config.checkpoint):
        checkpoint = config.checkpoint
        artifact = None
    else:
        api = wandb.Api()
        
        project = config.wandb_logger['project']
        artifact_name = config.checkpoint
        
        artifact = api.artifact(f"{project}/{artifact_name}", type='model')
        
        if len(artifact.manifest.entries) > 1:
            entry = artifact.get_entry('best_model.ckpt')
        elif len(artifact.manifest.entries) == 1:
            entry_name = next(iter(artifact.manifest.entries.keys()))
            entry = artifact.get_entry(entry_name)
        else:
            raise FileNotFoundError("Artifact does not contain checkpoint")
        
        checkpoint = entry.download()
    
    for data_fraction in args.data_fraction:
        logger.info(f"Loading data fraction {data_fraction}...")
        data = MIMICIIIDataModule.load_from_checkpoint(
            checkpoint,
            stage='finetune',
            data_fraction=data_fraction,
            data_config=config.data_config
        )
        module_config = FinetuneModuleConfig(config.module_config)
        model = FinetuneStrats.load_from_checkpoint(
            checkpoint,
            strict=True,
            logger=logger,
            module_config=module_config,
            features_info=data.get_features_info(),
        )
        offline = args.debug
        
        if not args.debug and not args.watch:
            model.compile(fullgraph=True)
        
        initial_state = copy_state(model)
        
        for fold in range(1, args.folds_number + 1):
            logger.info(f"Starting fold {fold}...")
            
            seed_everything(fold)
            model.load_state_dict(initial_state, strict=True)
            
            ##### INIT WANDB
            wandb_logger = WandbLogger(
                **config.wandb_logger,
                group=f"finetune_{data_fraction}",
                tags=[f"frac={data_fraction}", f"fold={fold}"],
                offline=offline,
                config=config,
                log_model=not offline,  # upload model checkpoint
                checkpoint_name=f"{config.name}_fold_{fold}_frac_{data_fraction}",
            )
            # wandb_logger.experiment.log({'fold': fold, 'data_fraction': data_fraction})
            # with open(config.ft_schedule) as f:
            #     ft_schedule = yaml.safe_load(f)
            
            wandb_logger.experiment.config.update({
                'fold': fold,
                'data_fraction': data_fraction,
                # 'ft_schedule': ft_schedule,
            }, allow_val_change=True)
            
            
            if args.watch:
                wandb_logger.watch(model, log='all')
            
            if artifact is not None and not offline:
                wandb_logger.experiment.use_artifact(artifact)
            
            yield fold, data_fraction, model, data, wandb_logger
            
            if args.watch:
                wandb_logger.experiment.unwatch(model)
                model._wandb_watch_called = False  # hack to avoid wandb validation error
            
            wandb_logger.experiment.finish()


def finetune_fold(
    run_name: str,
    config: MainConfig,
    model: FinetuneStrats,
    data: L.LightningDataModule,
    csv_logger: CSVLogger,
    wandb_logger: WandbLogger,
):
    # curves_logger_callback = CurvesLoggerCallback(csv_logger.log_dir)
    
    early_stop_callback = fts.FTSEarlyStopping(
        **config.early_stop_callback,
        verbose=True
    )
    checkpoint_callback = fts.FTSCheckpoint(
        **config.checkpoint_callback,
        filename=f'{run_name}/model-{{epoch}}-{{val_pr_roc_auc_sum:.2f}}',
        verbose=True,
    )
    
    scheduler = fts.FinetuningScheduler(ft_schedule=config.ft_schedule)
    
    
    trainer = L.Trainer(
        **config.trainer,
        logger=[wandb_logger],  # csv_logger
        num_sanity_val_steps=0,
        callbacks=[scheduler, early_stop_callback, checkpoint_callback],  # curves_logger_callback
        deterministic=True,
    )
    
    trainer.fit(
        model=model,
        datamodule=data,
    )
    trainer.test(  # TODO: implement test run only mode
        ckpt_path="best",
        datamodule=data,
    )

if __name__ == '__main__':
    main()
