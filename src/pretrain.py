import argparse
import logging

import wandb

from src.util.common import (
    create_data_module,
    create_model_module,
    create_trainer,
    create_wandb_logger,
    setup,
)
from src.util.config import (
    PretrainConfig,
    read_config,
)
from src.util.wandb import (
    DATA_MODULE,
    create_data_module_artifact,
    find_checkpoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--watch', action='store_true', help='Watch model with wandb')
    parser.add_argument('--runs-number', '-n', type=int, default=1, help='Number of the run')
    parser.add_argument('--dry-run', action='store_true', help='Dry run')
    args = parser.parse_args()
    config = read_config(args.config)
    
    if not isinstance(config, PretrainConfig):
        raise ValueError(f"Config stage must be 'pretrain', got {config.stage}")
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    setup()
    pretrain(args, config, logger)


def pretrain(args: argparse.Namespace, config: PretrainConfig, logger: logging.Logger):
    data_checkpoint, data_artifact = find_checkpoint(config, checkpoint_type=DATA_MODULE)
    data = create_data_module(config, logger, data_checkpoint)
    
    if data_artifact is None and not args.dry_run:
        logger.info("Creating data module artifact")
        # TODO: in a cluster environment, this may be executed multiple times
        #   Consider moving to data.prepare_data() method with prepare_data_per_node=False
        with wandb.init(
            config=config.model_dump(mode='json') | {'stage': DATA_MODULE},
            name=config.name,
            job_type=DATA_MODULE,
        ):
            checkpoint, data_artifact = create_data_module_artifact(config, data)
    for i in range(args.runs_number):
        try:
            wandb_logger = create_wandb_logger(
                args, config, [data_artifact] if data_artifact is not None else None,
                # args, config, [],
                config_extra={
                    'data_fraction': 1.0  # allows grouping by data_fraction in wandb dashboard
                })
            
            with wandb_logger.experiment:
                # everything logged inside this block will be logged to wandb
                model = create_model_module(config, logger, data)
                
                if args.watch:
                    wandb_logger.watch(model, log='all')
                
                if not args.debug and not args.watch and not args.dry_run:
                    model.compile(fullgraph=True, dynamic=True, mode="reduce-overhead")
                
                trainer = create_trainer(config, args, wandb_logger)
                
                trainer.fit(model=model, datamodule=data)
                
        except Exception:
            logger.exception(f"Error in fold {i + 1}")
            continue


if __name__ == '__main__':
    main()
