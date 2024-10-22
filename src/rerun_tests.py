import argparse
import hashlib
import logging
import os
import pickle
from typing import Literal

import lightning as L
import pandas as pd
import torch
import wandb
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
)

from src.lightning_module import (
    FinetuneModule,
    PretrainingModule,
)
from src.util.collator import Collator
from src.util.config_namespace import (
    MainConfig,
    read_config,
)
from src.util.data_module import (
    MIMICIIIDataModule,
    PersistentGPUDataLoader,
)
from src.util.dataset import (
    AbstractDataset,
    MemDataset,
)
from src.util.variable_scalers import (
    AbstractScaler,
    VariableECDFScaler,
    VariableStandardScaler,
)
from src.util.wandb import get_run_checkpoint


def rerun_finetune_tests(config, runs: list[wandb.apis.public.Run]):
    dataset, scalers = get_cached_dataset(config, get_run_checkpoint(runs[0]), split='test')
    num_features = len(scalers['variable_scaler'].variable_categories)
    data_loader = PersistentGPUDataLoader(
        dataset,
        batch_size=config.data_config['test']['batch_size'],
        shuffle=False,
        collate_fn=Collator(padding_value=num_features),
    )
    
    for run in runs:
        checkpoint = get_run_checkpoint(run)
        if run.config['stage'] == 'pretrain':
            model = PretrainingModule.load_from_checkpoint(
                checkpoint,
                strict=True,
                module_config=config.module_config,
                logger=logging.getLogger(__name__),
            )
        elif run.config['stage'] == 'finetune':
            model = FinetuneModule.load_from_checkpoint(
                checkpoint,
                strict=True,
                logger=logging.getLogger(__name__),
            )
        else:
            raise ValueError(f"Invalid stage: {run.config['stage']}")
        
        trainer = L.Trainer(
            **config.trainer,
            logger=False,
            num_sanity_val_steps=0,
        )
        result = trainer.test(
            model=model,
            dataloaders=data_loader,
        )
        # log results
        run.summary.update(result[0])


def recompute_pretrain_loss_summary(
    run: wandb.apis.public.Run,
    config: MainConfig,
    data_loader: DataLoader,
    scalers: dict[str, AbstractScaler],
    metric_prefix: str
):
    checkpoint = get_run_checkpoint(run)
    model = PretrainingModule.load_from_checkpoint(
        checkpoint,
        strict=True,
        module_config=config.module_config,
        logger=logging.getLogger(__name__),
    )
    
    trainer = L.Trainer(**config.trainer, barebones=True)
    
    val_losses = compute_pretrain_losses(data_loader, model, scalers, trainer)
    run.summary.update({f"{metric_prefix}_{k}": v for k, v in val_losses.items()})


def compute_pretrain_losses(data_loader, model, scalers, trainer):
    pred_true_mask = trainer.predict(model, data_loader)
    pred_true_mask: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    pred_stack = torch.vstack([pred for pred, true, mask in pred_true_mask]).to(torch.float32)
    true_stack = torch.vstack([true for pred, true, mask in pred_true_mask])  # float32
    mask_stack = torch.vstack([mask for pred, true, mask in pred_true_mask])  # bool
    events_df = pd.DataFrame({
        'true_value': true_stack.flatten().numpy(),
        'pred_value': pred_stack.flatten().numpy(),
        'variable': scalers['original'].variable_categories * true_stack.size(0),
        'mask': mask_stack.flatten().numpy(),
    })
    variable_type = pd.CategoricalDtype(categories=scalers['original'].variable_categories)
    events_df = events_df[events_df['mask']].astype({'variable': variable_type}).copy()
    
    def get_losses(df: pd.DataFrame, true_col: str, pred_col: str):
        diff = df[true_col] - df[pred_col]
        mse_loss = diff.pow(2).mean()
        mae_loss = diff.abs().mean()
        return mse_loss, mae_loss
    
    unscaled_events = scalers['original'].inverse_transform(events_df, value_col='true_value')
    unscaled_events = scalers['original'].inverse_transform(unscaled_events, value_col='pred_value')
    std_events = scalers['standard'].transform(unscaled_events, value_col='true_value')
    std_events = scalers['standard'].transform(std_events, value_col='pred_value')
    std_mse_loss, std_mae_loss = get_losses(std_events, 'true_value', 'pred_value')
    ecdf_events = scalers['ecdf'].transform(unscaled_events, value_col='true_value')
    ecdf_events = scalers['ecdf'].transform(ecdf_events, value_col='pred_value')
    ecdf_mse_loss, ecdf_mae_loss = get_losses(ecdf_events, 'true_value', 'pred_value')
    return {
        'ecdf_mse_loss': ecdf_mse_loss,
        'ecdf_mae_loss': ecdf_mae_loss,
        'standardized_mse_loss': std_mse_loss,
        'standardized_mae_loss': std_mae_loss,
    }


def get_cached_dataset(
    config: MainConfig,
    checkpoint: str,
    split: Literal['test', 'val'] = 'test',
    repeat_times=None
):
    if repeat_times is not None:
        # modify config to affect caching
        config.data_config[split]['repeat_times'] = repeat_times
    config_hash = hashlib.md5(str(sorted(((k, v) for k, v in config.data_config[split].items() if
                                          k != 'batch_size'))).encode()).hexdigest()
    dataset_key = f'/home/user/.cache/thesis/tmp/{split}_dataset-{config_hash}.pkl'
    # check if dataset is cached
    if os.path.exists(dataset_key):
        print(f"Loading cached dataset from {dataset_key}")
        with open(dataset_key, 'rb') as f:
            return pickle.load(f)
    
    print(f"Creating dataset for config {config.name}")
    data: MIMICIIIDataModule = MIMICIIIDataModule.load_from_checkpoint(
        checkpoint,
        stage=config.stage,
        data_config=config.data_config,
        logger=logging.getLogger(__name__),
    )
    
    dataset: AbstractDataset = getattr(data, f"{split}_dataset")
    
    dataset.load_data()
    dataset = ConcatDataset([dataset] * repeat_times)
    dataset = MemDataset(dataset)
    with open(dataset_key, 'wb') as f:
        print(f"Caching dataset to {dataset_key}")
        pickle.dump((dataset, data.scalers), f)
    return dataset, data.scalers


def compute_rescaled_pretraining_validation_loss(
    config: MainConfig,
    runs: list[wandb.apis.public.Run]
):
    checkpoint = get_run_checkpoint(runs[0])
    test_dataset, scalers = get_cached_dataset(config, checkpoint, split='test', repeat_times=50)
    val_dataset, scalers_1 = get_cached_dataset(config, checkpoint, split='val', repeat_times=50)
    assert scalers == scalers_1
    original_scaler: AbstractScaler = scalers['variable_scaler']
    standard_scaler = VariableStandardScaler()
    ecdf_scaler = VariableECDFScaler()
    variable_type = pd.CategoricalDtype(categories=original_scaler.variable_categories)
    train_events = pd.read_parquet(
        os.path.join(config.data_config['train']['path'], 'events.parquet'),
        columns=['value', 'variable']
    ).astype({'variable': variable_type})
    standard_scaler.fit(train_events)
    ecdf_scaler.fit(train_events)
    scalers = {
        'original': original_scaler,
        'standard': standard_scaler,
        'ecdf': ecdf_scaler,
    }
    collator = Collator(len(variable_type.categories))
    val_data_loader = PersistentGPUDataLoader(
        val_dataset,
        batch_size=config.data_config['val']['batch_size'],
        shuffle=False,
        collate_fn=collator,
    )
    test_data_loader = PersistentGPUDataLoader(
        test_dataset,
        batch_size=config.data_config['test']['batch_size'],
        shuffle=False,
        collate_fn=collator,
    )
    
    for run in runs:
        try:
            print(f"Recomputing pretraining loss summary for run {run.id}")
            recompute_pretrain_loss_summary(run, config, val_data_loader, scalers, 'best_val')
            recompute_pretrain_loss_summary(run, config, test_data_loader, scalers, 'test')
        except Exception as e:
            print(f"Error in run {run.id}: {e}")
            # get stacktrace
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    config = read_config(args.config)
    api = wandb.Api()
    runs = api.runs(config.wandb_logger['project'], filters={
        # '$or': [
        #     {"config.name": config.name},
        #     {"display_name": config.name},
        # ],
        "state": {"$ne": "running"},
        "display_name": config.name,
        "tags": {"$nin": ["archive", "with-bias"]},
        # "summary_metrics.test_epoch_ecdf_mae_loss": {"$exists": False},
        # "summary_metrics.test_auroc": {"$exists": False},
    })
    if not runs:
        print(f"No runs found for config {config.name}")
        return
    
    print(f"Found {len(runs)} runs for config {config.name}")
    
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if config.stage == 'pretrain':
        compute_rescaled_pretraining_validation_loss(config, runs)
    elif config.stage == 'finetune':
        rerun_finetune_tests(config, runs)


if __name__ == '__main__':
    main()
