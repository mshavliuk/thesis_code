from argparse import Namespace
from dataclasses import dataclass

from src.models.strats import StratsConfig
from src.util.dataset import DatasetConfig
from src.util.trainer import TrainerConfig

Namespace(
    load_ckpt_path=None,
    output_dir='../outputs/mimic_iii/pretrain/',
    seed=2023,
    lr=0.0005,
)


@dataclass
class Hyperparams:
    strats: StratsConfig
    dataset: DatasetConfig
    trainer: TrainerConfig
