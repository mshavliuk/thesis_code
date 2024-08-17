import argparse
import logging
from dataclasses import (
    dataclass,
    fields,
)
import lightning as L
import numpy as np
import torch
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
from lightning.pytorch.tuner import Tuner
from torch.utils.data import (
    DataLoader,
    Dataset as TorchDataset,
    Subset,
)

from src.models.lightning_module import (
    FinetuneStrats,
    ModuleConfig,
    OptimizerConfig,
    PretrainingStrats,
)
from src.util.callbacks import (
    CurvesLoggerCallback,
    FoldInfoLoggerCallback,
)
from src.util.collator import Collator
from src.util.data_module import MIMICIIIDataModule
from src.util.dataset import (
    AbstractDataset,
    DatasetConfig,
    FinetuneDataset,
    PretrainDataset,
)


@dataclass
class Splits[T]:
    train: T
    val: T
    test: T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--stage', type=str, required=True, help='Training stage')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--folds_number', type=int, default=1, help='Number of folds')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of data to use')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    logging.basicConfig(level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    if not args.debug:
        print('Enter run name: ', end='')
        run_name = input()
    else:
        run_name = 'debug'

    torch.set_float32_matmul_precision('medium')
    if args.stage == 'pretrain':
        pretrain(config, logger, args, run_name)
    elif args.stage == 'finetune':
        finetune(config, logger, args, run_name)


def pretrain(hparams: dict, logger: logging.Logger, args: argparse.Namespace, run_name: str):
    seed_everything(42)
    csv_logger = CSVLogger(hparams['results_dir'], name='pretrain', version=run_name)
    
    # datasets = get_datasets(hparams, PretrainDataset)

    if args.checkpoint:
        model = PretrainingStrats.load_from_checkpoint(
            args.checkpoint, logger=logger, strict=True)
        data = MIMICIIIDataModule.load_from_checkpoint(args.checkpoint)
    else:
        data = MIMICIIIDataModule(
            stage='pretrain',
            data_config=hparams['data_module'],
        )
        model_hparams = ModuleConfig(
            model={
                **hparams['model'],
                # get the number of variables and demographics from the dataset
                **data.get_variables_metadata()
            },
            optimizer=hparams['optimizer'],
        )
        model = PretrainingStrats(model_hparams)
    
    # datasets.train.load_data()
    # scalers = datasets.train.get_scalers()
    #
    # for split in 'val', 'test':
    #     dataset = getattr(datasets, split)
    #     dataset.set_scalers(scalers)
    #     dataset.load_data()
    
    # dataloaders = get_data_loaders(datasets, hparams)
    early_stop = EarlyStopping(
        **hparams['early_stop_callback'],
        verbose=True
    )
    checkpoint = ModelCheckpoint(
        **hparams['checkpoint_callback'],
        filename='strats-{epoch}-{val_epoch_loss:.2f}',
        verbose=True
    )
    
    wandb_logger = WandbLogger(project="Strats", offline=args.debug, name=run_name)
    wandb_logger.experiment.tags = ["pretrain"]
    
    trainer = L.Trainer(
        precision='bf16-mixed',
        # accumulate_grad_batches=2,
        # accelerator='cpu', # FIXME: remove
        **hparams['trainer'],
        logger=[wandb_logger, csv_logger],
        deterministic=True,
        callbacks=[early_stop, checkpoint],
    )
    if not args.debug:
        model.compile(fullgraph=True)
    
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model, datamodule=data)
    # print(lr_finder.results)
    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    
    trainer.fit(
        model=model,
        datamodule=data,
        
        # train_dataloaders=dataloaders.train,
        # val_dataloaders=dataloaders.val,
    )
    trainer.test(
        ckpt_path="best",
        datamodule=data,
    )
    return model


def finetune(hparams: dict, logger: logging.Logger, args: argparse.Namespace, run_name: str):
    csv_logger = CSVLogger(
        hparams['results_dir'],
        name='finetune',
        version=run_name
    )
    optimizer_conf = OptimizerConfig(**hparams['optimizer'])
    model = FinetuneStrats.load_from_checkpoint(
        args.checkpoint, strict=True, logger=logger, optimizer=optimizer_conf
    )
    data = MIMICIIIDataModule.load_from_checkpoint(
        args.checkpoint, stage='finetune', data_fraction=args.data_fraction)

    if not args.debug:
        model.compile(fullgraph=True)
    
    # keep a state copy to reset the model for each fold
    pretrain_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    for fold in range(1, args.folds_number + 1):
        print(f"Training fold {fold}...")
        # TODO: try to freeze the pretrained backbone and train the head only
        # TODO: try to swap entire head with a new one
        model.load_state_dict(pretrain_state, strict=True)
        
        seed_everything(fold)
        finetune_fold(args, fold, hparams, model, data, run_name, csv_logger)


def hypertuning():
    raise NotImplementedError
    
    # TODO: Trainer(accumulate_grad_batches=X)
    # From https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html:
    # TODO: lr scheduler
    # TODO: Enable Stochastic Weight Averaging using the callback
    #   trainer = Trainer(callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])
    # TODO: batch size finder
    #   tuner = Tuner(trainer)
    #   # Auto-scale batch size by growing it exponentially (default)
    #   tuner.scale_batch_size(model, mode="power")
    #   # Auto-scale batch size with binary search
    #   tuner.scale_batch_size(model, mode="binsearch")


def finetune_fold(
    args: argparse.Namespace,
    fold: int,
    hparams: dict,
    model: FinetuneStrats,
    data: L.LightningDataModule,
    run_name: str,
    csv_logger: CSVLogger
):
    fold_run_name = f"{run_name}_frac_{args.data_fraction}_fold_{fold}"
    fold_info_callback = FoldInfoLoggerCallback(fold, args.data_fraction)
    curves_logger_callback = CurvesLoggerCallback(csv_logger.log_dir)
    
    early_stop_callback = EarlyStopping(
        **hparams['early_stop_callback'],
        verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        **hparams['checkpoint_callback'],
        filename=f'{fold_run_name}/model-{{epoch}}-{{val_pr_roc_auc_sum:.2f}}',
        verbose=True
    )
    
    wandb_logger = WandbLogger(
        name=run_name, project="Strats", group=f"finetune_{args.data_fraction}", offline=args.debug)
    wandb_logger.experiment.tags = ["finetune", f"frac={args.data_fraction}", f"fold={fold}"]
    
    trainer = L.Trainer(
        **hparams['trainer'],
        logger=[wandb_logger, csv_logger],
        num_sanity_val_steps=0,
        callbacks=[early_stop_callback, checkpoint_callback, fold_info_callback,
                   curves_logger_callback],
        deterministic=True,
    )
    
    trainer.fit(
        model=model,
        datamodule=data,
    )
    trainer.test(
        ckpt_path="best",
        datamodule=data,
    )
    
    if not args.debug:
        best_model_checkpoint_path = checkpoint_callback.best_model_path
        wandb_logger.experiment.log_artifact(
            best_model_checkpoint_path, type='checkpoint', name=f"fold_{fold}_best")
        wandb_logger.experiment.finish()


if __name__ == '__main__':
    main()
