import logging
import os
import sys

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger

from src.models.strats import (
    Strats,
    StratsConfig,
)
from src.util.collator import Collator
from src.util.data_loader import DataLoader
from src.util.dataset import (
    Dataset,
    DatasetConfig,
)
# from util.trainer import (
#     Trainer,
#     TrainerConfig,
# )


def main():
    # model = ModelBuilder(config).build()
    
    seed_everything(42)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    logger.info("Loading training data...")
    training_data = Dataset(logging, DatasetConfig(
        path='/home/user/.cache/thesis/data/preprocessed/train',
        variables_dropout=0.2,
    ))
    logger.info("Loading validation data...")
    val_data = Dataset(logging, DatasetConfig(
        path='/home/user/.cache/thesis/data/preprocessed/val',
        variables_dropout=0,
    ))
    train_loader = DataLoader(
        training_data,
        # num_workers=4,
        device=device,
        batch_size=16, shuffle=True, collate_fn=Collator())
    val_loader = DataLoader(
        val_data,
        # num_workers=4,
        device=device,
        batch_size=32, shuffle=False, collate_fn=Collator())
    
    model = Strats(StratsConfig(
        demographics_num=training_data.num_demographics,
        features_num=training_data.num_variables,
        hid_dim=64,
        num_layers=2,
        num_heads=16,
        dropout=0.2,
        attention_dropout=0.2,
        head='forecast'
    ))
    logger.info(f"Model: {model}")
    logger.info("Moving model to device...")
    model = model.to(device)
    logger.info("Compiling model...")
    if os.environ.get('JETBRAINS_REMOTE_RUN') or '.pycharm_helpers' in sys.argv[0]:
        ...
    else:
        model = torch.compile(model)
    # if config.hyperparams:
    #     hyperparams = config.hyperparams
    # else:
    #     tuner = Tuner(config, logger)
    #     hyperparams = tuner.get_best_hyperparams(model, dataset)
    # trainer = Trainer(logger)
    # trainer_config = TrainerConfig(
    #     train_batch_size=16,
    #     eval_batch_size=32,
    #     lr=5e-4,
    #     max_epochs=30
    # )
    # trainer.train(trainer_config, model, train_loader, val_loader)
    
    wandb_logger = WandbLogger(project="Strats")
    trainer = Trainer(logger=wandb_logger)
    trainer


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # TODO: decide whether to get configs as json/yaml or as command line arguments
    # parser.add_argument('--config', type=str, required=True, help='Path to config file')
    # args = parser.parse_args()
    # config = Config(args.config)
    logging.basicConfig(level=logging.INFO)
    main()
