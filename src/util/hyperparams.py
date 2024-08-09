from argparse import Namespace
from dataclasses import dataclass

from src.models.strats import StratsConfig
from src.util.dataset import DatasetConfig
from src.util.trainer import TrainerConfig

Namespace(
    pos_class_weight=1.0,  # (num_train-num_train_pos)/num_train_pos
    
    # running related
    run='1o10',
    model_type='strats',
    load_ckpt_path=None,
    output_dir='../outputs/mimic_iii/pretrain/',
    output_dir_prefix='',
    seed=2023,
    max_epochs=30,
    patience=10,
    
    lr=0.0005,
    gradient_accumulation_steps=1,
    print_train_loss_every=100,
    validate_after=-1,
    validate_every=None,
    
    # model related
    weights=None,  # or path to weights, or weight? or initialize externally?
)


@dataclass
class Hyperparams:
    strats: StratsConfig
    dataset: DatasetConfig
    trainer: TrainerConfig
