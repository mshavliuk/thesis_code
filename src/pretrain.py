import argparse
import logging

import torch

from src.util.common import (
    find_checkpoint,
    create_data_module,
    create_model_module,
    create_trainer,
    create_wandb_logger,
)
from src.util.config_namespace import (
    MainConfig,
    read_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--watch', action='store_true', help='Watch model with wandb')
    parser.add_argument('--runs-number', '-n', type=int, default=1, help='Number of the run')
    parser.add_argument('--from-run', type=str, default=None, help='Run to load checkpoint from')
    args = parser.parse_args()
    config = read_config(args.config)
    
    if config.stage != 'pretrain':
        raise ValueError(f"Config stage must be 'pretrain', got {config.stage}")
    
    logging.basicConfig(level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # FLOAT16
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True  # BFLOAT16
    
    pretrain(args, config, logger)


def pretrain(args: argparse.Namespace, config: MainConfig, logger: logging.Logger):
    checkpoint, artifact = find_checkpoint(config, args, return_artifact=True)
    data = create_data_module(config, logger, checkpoint)
    #  Add try catch
    
    for i in range(args.runs_number):
        model = create_model_module(config, logger, data, checkpoint)
        wandb_logger = create_wandb_logger(args, config, artifact)
        
        if args.watch:
            wandb_logger.watch(model, log='all')
        
        if not args.debug and not args.watch:
            model.compile(fullgraph=True, dynamic=True, mode="reduce-overhead")
        
        trainer = create_trainer(args, config, wandb_logger)
        
        trainer.fit(model=model, datamodule=data, ckpt_path=checkpoint)
        
        wandb_logger.experiment.finish()


if __name__ == '__main__':
    main()
