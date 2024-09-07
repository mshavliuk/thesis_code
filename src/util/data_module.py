from dataclasses import (
    dataclass,
    fields,
)
from typing import (
    Any,
    Literal,
)

import lightning as L
import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Subset,
    WeightedRandomSampler,
)

from src.models.strats import FeaturesInfo
from src.util.collator import Collator
from src.util.config_namespace import ConfigNamespace
from src.util.dataset import (
    AbstractDataset,
    AbstractScaler,
    DatasetConfig,
    FinetuneDataset,
    MemDataset,
    PretrainDataset,
)


@dataclass(frozen=True)
class DataLoaderConfig(DatasetConfig):
    batch_size: int
    balanced: bool
    repeat_times: int
    
    @property
    def dataset_config(self) -> DatasetConfig:
        return DatasetConfig(**{f.name: getattr(self, f.name) for f in fields(DatasetConfig)})


class DataModuleConfig(ConfigNamespace):
    bootstrap: bool = False
    train: DataLoaderConfig
    val: DataLoaderConfig
    test: DataLoaderConfig


class MIMICIIIDataModule(L.LightningDataModule):
    def __init__(
        self,
        stage: Literal['pretrain', 'finetune'],
        data_config: dict[str, dict],
        data_fraction: float = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        data_config = DataModuleConfig(**data_config)
        self.hparams.data_config = data_config
        
        self.scalers: dict[str, AbstractScaler] | None = None
        self.setups_completed = set()
        
        if stage == 'pretrain':
            self.train_dataset = PretrainDataset(data_config.train.dataset_config)
            self.val_dataset = PretrainDataset(data_config.val.dataset_config)
            self.test_dataset = PretrainDataset(data_config.test.dataset_config)
        elif stage == 'finetune':
            self.train_dataset = FinetuneDataset(data_config.train.dataset_config)
            self.val_dataset = FinetuneDataset(data_config.val.dataset_config)
            self.test_dataset = FinetuneDataset(data_config.test.dataset_config)
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def get_subset(self, dataset: AbstractDataset, fraction: float) -> Subset:
        idx = np.random.choice(dataset.indexes, int(len(dataset) * fraction),
                                   replace=self.hparams.data_config.bootstrap)
        return Subset(dataset, idx)
    
    def get_features_info(self) -> FeaturesInfo:
        return self.train_dataset.get_features_info()
    
    def setup(self, stage: Literal['fit', 'test']):
        if stage in self.setups_completed:
            return
        
        if stage == 'fit':
            self.train_dataset.load_data()  # TODO: rename to setup?
            self.scalers = self.train_dataset.get_scalers()
            self.val_dataset.set_scalers(self.scalers)
            self.val_dataset.load_data()
        
        if stage == 'test':
            self.test_dataset.set_scalers(self.scalers)
            self.test_dataset.load_data()
            dataset = ConcatDataset([self.test_dataset] * \
                                    self.hparams.data_config.test.repeat_times)
            # the test dataset has to be put in memory for consistent results
            # one for every fold
            self.test_dataset = MemDataset(dataset)
        
        self.setups_completed.add(stage)
    
    def state_dict(self) -> dict[str, Any]:
        return {'scalers': self.scalers}
    
    def load_state_dict(self, state_dict) -> None:
        if self.scalers != state_dict['scalers']:
            self.scalers = state_dict['scalers']
            self.train_dataset.set_scalers(self.scalers)
            self.val_dataset.set_scalers(self.scalers)
            self.test_dataset.set_scalers(self.scalers)
    
    def train_dataloader(self):
        if self.hparams.data_config.train.repeat_times != 1:
            raise NotImplementedError("Train repeat times is not supported")
        
        batch_size = self.hparams.data_config.train.batch_size
        
        if self.hparams.data_fraction is None:
            dataset = self.train_dataset
        else:
            dataset = self.get_subset(self.train_dataset, self.hparams.data_fraction)
        
        if self.hparams.data_config.train.balanced:
            labels = np.array([i['label'][0] for i in dataset], dtype=np.int8)
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts / len(class_counts)
            sample_weights = class_weights[labels]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=max(len(dataset), batch_size),
                replacement=True, # has to be true, otherwise it samples every element exactly once
            )
        else:
            sampler = None
            
        return DataLoader(
            dataset,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=Collator(),
            batch_size=batch_size)
    
    def val_dataloader(self):
        if self.hparams.data_fraction is None:
            dataset = self.val_dataset
        else:
            dataset = self.get_subset(self.val_dataset, self.hparams.data_fraction)
        
        # every time (for every fold) the new subset is returned
        dataset = ConcatDataset([dataset] * self.hparams.data_config.val.repeat_times)
        dataset = MemDataset(dataset)
        
        return DataLoader(
            dataset,
            batch_size=self.hparams.data_config.val.batch_size,
            shuffle=False,
            collate_fn=Collator()
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.data_config.test.batch_size,
            shuffle=False,
            collate_fn=Collator()
        )
