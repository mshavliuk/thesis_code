import argparse
import io
import logging

import torch

from src.util.common import (
    create_data_module,
    create_model_module,
    create_trainer,
    create_wandb_logger,
    find_checkpoint,
)
from src.util.config_namespace import (
    MainConfig,
    read_config,
)


def finetune(
    args: argparse.Namespace, config: MainConfig, logger
):
    checkpoint, artifact = find_checkpoint(config, args, return_artifact=True)
    data = create_data_module(config, logger, checkpoint)
    model = create_model_module(config, logger, data, checkpoint)
    
    initial_state = copy_state(model)
    
    if not args.debug and not args.watch:
        model.compile(fullgraph=True, dynamic=True)
    
    for data_fraction in args.data_fraction:
        with data.folds(args.folds_number, data_fraction) as folds:
            for fold_index in folds:
                fold_number = fold_index + 1
                logger.info(f"Starting fold {fold_number} with data fraction {data_fraction}...")
                try:
                    model.load_state_dict(initial_state, strict=True)
                    torch.cuda.synchronize()
                    finetune_fold(args, artifact, config, data, data_fraction, fold_number, model)
                except Exception as e:
                    logger.exception(f"Error in fold {fold_number}: {e}")


def finetune_fold(args, artifact, config, data, data_fraction, fold_number, model):
    wandb_logger = create_wandb_logger(args, config, artifact)
    with wandb_logger.experiment as run:
        run.config.update({
            'fold': fold_number,
            'data_fraction': data_fraction,
        }, allow_val_change=True)
        if args.watch:
            wandb_logger.watch(model, log='all')
        
        trainer = create_trainer(args, config, wandb_logger)
        trainer.fit(
            model=model,
            datamodule=data,
        )
        trainer.test(
            ckpt_path="best",
            datamodule=data,
        )
        
        if args.watch:
            run.unwatch(model)
            model._wandb_watch_called = False  # hack to avoid wandb validation error


def copy_state(model: torch.nn.Module) -> dict:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    parser.add_argument('--debug', '-d', action='store_true', help='Debug mode')
    parser.add_argument('--watch', '-w', action='store_true', help='Watch model with wandb')
    parser.add_argument('--folds_number', '-n', type=int, default=1, help='Number of folds')
    parser.add_argument('--data_fraction', '-f',
                        type=lambda line: tuple(float(v) for v in (line.split(','))),
                        default='1.0',
                        help='Fraction of data to use')
    args = parser.parse_args()
    config = read_config(args.config)
    
    if config.stage != 'finetune':
        raise ValueError(f"Config stage must be 'finetune', got {config.stage}")
    
    logging.basicConfig(level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    
    finetune(args, config, logger)


if __name__ == '__main__':
    main()
