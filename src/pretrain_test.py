import argparse
import hashlib
import logging
import os
import pickle
from itertools import (
    pairwise,
)
from pathlib import Path
from typing import (
    Literal,
)

import pandas as pd
import torch
import wandb
from torch.utils.data import (
    ConcatDataset,
)

from src.util.collator import Collator
from src.util.common import (
    create_data_module,
    create_model_module,
    create_trainer,
    setup,
)
from src.util.config import (
    FinetuneConfig,
    MainConfig,
    PretrainConfig,
    read_config,
)
from src.util.data_module import (
    PersistentGPUDataLoader,
    SplitConfig,
)
from src.util.dataset import (
    AbstractDataset,
    MemDataset,
)
from src.util.variable_scalers import (
    VariableECDFScaler,
    VariableStandardScaler,
)
from src.util.wandb import (
    find_checkpoint,
    get_run_checkpoint,
)


def rerun_finetune_tests(runs: list[wandb.apis.public.Run], config: FinetuneConfig):
    data_checkpoint, data_artifact = find_checkpoint(config, checkpoint_type='data_module')
    logger = logging.getLogger(__name__)
    
    data = create_data_module(config, logger, data_checkpoint)
    # risk_dist_path = Path(os.environ['DATA_DIR'] + '/plots/risk_score_distributions/').resolve()
    # risk_dist_path.mkdir(parents=True, exist_ok=True)
    # curves_callback = CurvesLoggerCallback(os.environ['DATA_DIR'] + '/plots/curves/')
    # config.trainer['callbacks'] = [curves_callback]
    trainer = create_trainer(config)
    for run in runs:
        logger.info(f"Recomputing finetune loss summary for run {run.id}")
        # curves_callback.set_filename_prefix(f"{run.name}_run_{run.id}_")
        checkpoint = get_run_checkpoint(run, type='model')
        model = create_model_module(config, logger, data, checkpoint)
        
        # preds = trainer.predict(model, data.test_dataloader())
        # preds = torch.sigmoid(torch.vstack(preds).squeeze())
        #
        # truths = torch.vstack([b['label'] for b in data.test_dataloader()]).squeeze()
        # df = pd.DataFrame({
        #     'risk_score': preds.float().numpy(),
        #     'label': truths.bool().cpu().numpy(),
        # })
        # # plot distributions of risk scores by class
        # fig, ax = plt.subplots()
        # sns.kdeplot(data=df, x='risk_score', hue='label', fill=True, ax=ax, clip=(0, 1), common_norm=False)
        # fig.tight_layout()
        # fig.show()
        # fig.savefig(risk_dist_path / f"{run.name}_{run.id}.png")
        #
        
        # exit()
        default_results = trainer.test(
            model=model,
            datamodule=data,
        )[0]
        
        with model.with_weighted_loss(data.train_dataloader()):
            weighted_result = trainer.test(
                model=model,
                datamodule=data,
            )[0]
        
        with model.with_unweighted_loss():
            unweighted_result = trainer.test(
                model=model,
                datamodule=data,
            )[0]
        
        weighted_loss = weighted_result.pop('test_loss')
        unweighted_loss = unweighted_result.pop('test_loss')
        test_loss = default_results.pop('test_loss')
        # check if all but loss metrics are equal
        for m1, m2 in pairwise((default_results, weighted_result, unweighted_result)):
            if m1 != m2:
                raise ValueError(f"Results are not equal: {m1} != {m2}")
        
        run.summary.update({
            'unweighted_test_loss': unweighted_loss,
            'weighted_test_loss': weighted_loss,
            'test_loss': test_loss,
        })


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
    
    losses = {}
    
    for scaler_name, scaler in filter(lambda x: x[0] != 'original', scalers.items()):
        scaled_events = scaler.transform(unscaled_events, value_col='true_value')
        scaled_events = scaler.transform(scaled_events, value_col='pred_value')
        mse_loss, mae_loss = get_losses(scaled_events, 'true_value', 'pred_value')
        losses[f'{scaler_name}_mse_loss'] = mse_loss.item()
        losses[f'{scaler_name}_mae_loss'] = mae_loss.item()
    
    return losses


def get_cached_dataset(
    config: MainConfig,
    checkpoint: str,
    split: Literal['test', 'val'] = 'test',
    repeat_times=None
):
    cfg: SplitConfig = getattr(config.data_config, split)
    if repeat_times is not None:
        # modify config to affect caching
        cfg.loader.repeat_times = repeat_times
    config_hash = hashlib.md5(cfg.__pydantic_serializer__.to_json(cfg)).hexdigest()
    dataset_path = Path(os.environ['TEMP_DIR'], f"{split}_dataset-{config_hash}.pkl")
    # check if dataset is cached
    
    if dataset_path.exists():
        print(f"Loading cached dataset from {dataset_path}")
        with open(dataset_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Creating dataset for config {config.name}")
    
    data = create_data_module(config, logging.getLogger(__name__), checkpoint)
    
    dataset: AbstractDataset = getattr(data, f"{split}_dataset")
    
    dataset.load_data()
    dataset = ConcatDataset([dataset] * repeat_times)
    dataset = MemDataset(dataset)
    
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open('wb') as f:
        print(f"Caching dataset to {dataset_path}")
        pickle.dump((dataset, data.scalers), f)
    return dataset, data.scalers


def compute_pretraining_loss(
    runs: list[wandb.apis.public.Run],
    config: PretrainConfig,
):
    data_checkpoint, data_artifact = find_checkpoint(config, checkpoint_type='data_module')
    test_dataset, scalers = get_cached_dataset(config,
                                               data_checkpoint,
                                               split='test',
                                               repeat_times=50)
    val_dataset, scalers_1 = get_cached_dataset(config,
                                                data_checkpoint,
                                                split='val',
                                                repeat_times=50)
    assert scalers == scalers_1
    
    original_scaler = scalers['variable_scaler']
    
    variable_type = pd.CategoricalDtype(categories=original_scaler.variable_categories)
    train_events = pd.read_parquet(
        config.data_config.train.dataset.path / 'events.parquet',
        columns=['value', 'variable']
    ).astype({'variable': variable_type})
    
    if isinstance(original_scaler, VariableECDFScaler):
        ecdf_scaler = original_scaler
        standard_scaler = VariableStandardScaler()
        standard_scaler.fit(train_events)
    elif isinstance(original_scaler, VariableStandardScaler):
        standard_scaler = original_scaler
        ecdf_scaler = VariableECDFScaler()
        ecdf_scaler.fit(train_events)
    else:
        raise NotImplementedError(f"Unexpected scaler type: {type(original_scaler)}")
    
    scalers = {
        'original': original_scaler,
        'standard': standard_scaler,
        'ecdf': ecdf_scaler,
    }
    
    collator = Collator(len(variable_type.categories))
    val_data_loader = PersistentGPUDataLoader(
        val_dataset,
        batch_size=config.data_config.val.loader.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_data_loader = PersistentGPUDataLoader(
        test_dataset,
        batch_size=config.data_config.test.loader.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    
    trainer = create_trainer(config)
    
    for run in runs:
        try:
            print(f"Recomputing pretraining loss summary for run {run.id}")
            
            checkpoint = get_run_checkpoint(run, type='model')
            model = create_model_module(config, logging.getLogger(__name__), None, checkpoint)
            
            val_losses = compute_pretrain_losses(val_data_loader, model, scalers, trainer)
            
            test_losses = compute_pretrain_losses(test_data_loader, model, scalers, trainer)
            run.summary.update(
                {f"test_{k}": v for k, v in test_losses.items()} |
                {f"best_val_{k}": v for k, v in val_losses.items()})
        
        except Exception as e:
            print(f"Error in run {run.id}: {e}")
            # get stacktrace
            import traceback
            traceback.print_exc()


def mark_best_run_artifact(runs: list[wandb.apis.public.Run], metric: str):
    best_run = min(runs, key=lambda r: r.summary[metric])
    
    # mark best run artifact as best
    best_artifact: wandb.Artifact = next(
        artifact for artifact in best_run.logged_artifacts() if
        artifact.type == 'model')
    best_artifact.aliases += ['best']
    best_artifact.save()
    print(f"Best run's ({best_run.id}) artifact was marked as 'best'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    config = read_config(args.config)
    api = wandb.Api()
    runs = api.runs(filters={
        "state": "finished",
        "display_name": config.name,
        "tags": {"$nin": ["archive", "with-bias"]},
        "config.stage": config.stage,
        'config.data_fraction': {'$in': [0.1, 0.5, 1.0]}, # FIXME: remove this
        '$or': [
            {'summary_metrics.weighted_test_loss': {'$exists': False}},
            {'summary_metrics.unweighted_test_loss': {'$exists': False}},
            {'summary_metrics.test_loss': {'$exists': False}},
        ],
    })
    if not runs:
        print(f"No runs found for config {config.name}")
        return
    
    print(f"Found {len(runs)} runs for config {config.name}")
    
    setup()
    
    if isinstance(config, PretrainConfig):
        compute_pretraining_loss(runs, config)
        
        scaler_type = {
            VariableECDFScaler.__name__: 'ecdf',
            VariableStandardScaler.__name__: 'standard',
        }[config.data_config.val.dataset.scaler_class]
        
        mark_best_run_artifact(runs, f"best_val_{scaler_type}_mse_loss")
    elif isinstance(config, FinetuneConfig):
        rerun_finetune_tests(runs, config)


if __name__ == '__main__':
    main()
