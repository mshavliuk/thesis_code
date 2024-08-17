from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
)

import lightning as L
import numpy as np
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Subset,
)

from src.util.collator import Collator
from src.util.dataset import (
    AbstractDataset,
    AbstractScaler,
    DatasetConfig,
    FinetuneDataset,
    PretrainDataset,
)


@dataclass
class Splits[T]:
    train: T
    val: T
    test: T


class MIMICIIIDataModule(L.LightningDataModule):
    scalers: dict[str, AbstractScaler] = None
    
    def __init__(
        self,
        stage: Literal['pretrain', 'finetune'],
        data_config: dict[str, dict],
        batch_size: int = None,  # allows using batch_size finder
        data_fraction: float = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        data_config = deepcopy(data_config)  # avoid modifying the original dict
        self.train_batch_size = data_config['train'].pop('batch_size')
        self.val_batch_size = data_config['val'].pop('batch_size')
        self.test_batch_size = data_config['test'].pop('batch_size')
        
        if stage == 'pretrain':
            self.train_dataset = PretrainDataset(DatasetConfig(**data_config['train']))
            self.val_dataset = PretrainDataset(DatasetConfig(**data_config['val']))
            self.test_dataset = PretrainDataset(DatasetConfig(**data_config['test']))
        elif stage == 'finetune':
            self.train_dataset = FinetuneDataset(DatasetConfig(**data_config['train']))
            self.val_dataset = FinetuneDataset(DatasetConfig(**data_config['val']))
            self.test_dataset = FinetuneDataset(DatasetConfig(**data_config['test']))
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
    def get_subset(self, dataset: AbstractDataset, fraction: float) -> Subset:
        # TODO: think if replace=True (bootstrap) is better
        # indices = np.random.choice(len(dataset), int(len(dataset) * fraction), replace=False)
        indices = np.random.choice(len(dataset), int(len(dataset) * fraction), replace=True)
        return Subset(dataset, indices)
    
    def get_variables_metadata(self):
        return self.train_dataset.get_variables_metadata()
    
    def setup(self, stage: Literal['fit', 'test']):
        if stage == 'fit':
            if self.scalers is not None:
                self.train_dataset.set_scalers(self.scalers)
            self.train_dataset.load_data()  # TODO: rename to setup?
            self.scalers = self.train_dataset.get_scalers()
            self.val_dataset.set_scalers(self.scalers)
            self.val_dataset.load_data()
        
        if stage == 'test':
            # scalers = ... # expect being loaded
            self.test_dataset.set_scalers(self.scalers)
            self.test_dataset.load_data()
    
    def state_dict(self) -> dict[str, Any]:
        return {'scalers': self.scalers}
    
    def load_state_dict(self, state_dict) -> None:
        self.scalers = state_dict['scalers']
    
    def train_dataloader(self):
        if self.hparams.data_fraction is None:
            dataset = self.train_dataset
        else:
            dataset = self.get_subset(self.train_dataset, self.hparams.data_fraction)
        return DataLoader(
            dataset,
            drop_last=True,
            shuffle=True,
            collate_fn=Collator(),
            batch_size=self.hparams.batch_size or self.train_batch_size)
    
    def val_dataloader(self):
        if self.hparams.data_fraction is None:
            dataset = self.val_dataset
        else:
            dataset = self.get_subset(self.val_dataset, self.hparams.data_fraction)
            
        dataset = ConcatDataset([dataset] * 3)
        return DataLoader(
            dataset,
            drop_last=True,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=Collator()
        )
    
    def test_dataloader(self):
        # always use entire dataset
        return DataLoader(
            self.test_dataset,
            drop_last=True,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=Collator()
        )
