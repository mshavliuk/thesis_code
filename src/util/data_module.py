import contextlib
import logging
from typing import (
    Generator,
    Literal,
)

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from pydantic import (
    BaseModel,
    Field,
)
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Subset,
    WeightedRandomSampler,
)

from src.models.strats_original import FeaturesInfo
from src.util.collator import Collator
from src.util.dataset import (
    AbstractDataset,
    FinetuneDataset,
    FinetuneDatasetConfig,
    MemDataset,
    PretrainDataset,
    PretrainDatasetConfig,
)
from src.util.variable_scalers import AbstractScaler


class DataLoaderConfig(BaseModel, extra='forbid'):
    batch_size: int = Field(gt=0)
    repeat_times: int = Field(ge=1)


class FinetuneDataLoaderConfig(DataLoaderConfig, extra='forbid'):
    balanced: bool = False


class SplitConfig(BaseModel, extra='forbid'):
    loader: DataLoaderConfig
    dataset: PretrainDatasetConfig


class DataModuleConfig(BaseModel, extra='forbid'):
    train: SplitConfig
    val: SplitConfig
    test: SplitConfig


class FinetuneSplitConfig(BaseModel, extra='forbid'):
    loader: FinetuneDataLoaderConfig
    dataset: FinetuneDatasetConfig | None = None


class FinetuneDataModuleConfig(BaseModel, extra='forbid'):
    train: FinetuneSplitConfig
    val: FinetuneSplitConfig
    test: FinetuneSplitConfig


class PersistentGPUDataLoader(DataLoader):
    data = {}  # shared between all instances
    
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.dataset_key = str(hash(dataset))
        self.data[self.dataset_key] = []
        for batch in self._get_iterator():
            self.data[self.dataset_key].append(
                {k: v.cuda(non_blocking=True) for k, v in batch.items()})
        torch.cuda.synchronize()
    
    def __iter__(self):
        yield from iter(self.data[self.dataset_key])
    
    def clear(self):
        if self.dataset_key in self.data:
            del self.data[self.dataset_key]


class MIMICIIIDataModule(L.LightningDataModule):
    
    
    def __init__(
        self,
        stage: Literal['pretrain', 'finetune'],
        data_config: DataModuleConfig | dict,
        logger: logging.Logger,
        new_config: FinetuneDataModuleConfig | None = None
    ):
        super().__init__()
        self._logger = logger
        self.scalers: dict[str, AbstractScaler] | None = None
        self.setups_completed = set()
        self.val_dataloaders = []
        
        if isinstance(data_config, dict):
            data_config = DataModuleConfig(**data_config)
        
        if stage == 'pretrain':
            dataset_class = PretrainDataset
            
            if new_config is not None:
                raise NotImplementedError("New data_config is not supported for pretrain stage")
        
        elif stage == 'finetune':
            if new_config is None:
                raise ValueError("New data_config has to be provided for finetune stage")
            
            for split_name, split_config in data_config.model_dump().items():
                new_split_config = getattr(new_config, split_name)
                if new_split_config.dataset is None:
                    new_split_config.dataset = FinetuneDatasetConfig(**split_config['dataset'])
            
            data_config = new_config
            dataset_class = FinetuneDataset
        
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        self.save_hyperparameters(
            {'data_config': data_config.model_dump(mode='json')},
            logger=False
        )
        self.config = data_config
        
        self._orig_train_dataset = dataset_class(data_config.train.dataset)
        self._orig_val_dataset = dataset_class(data_config.val.dataset)
        self._orig_test_dataset = dataset_class(data_config.test.dataset)
        self.train_dataset = self._orig_train_dataset
        self.val_dataset = self._orig_val_dataset
        self.test_dataset = self._orig_test_dataset
        
        self.collator = Collator(self.train_dataset.num_variables)
    
    def prepare_data(self):
        pass
    
    @contextlib.contextmanager
    def folds(self, folds_number: int, data_fraction: float) -> Generator[
        Generator[int, None, None], None, None]:
        def _iterate_folds():
            for fold_index in range(folds_number):
                self.train_dataset = self._subset(
                    self._orig_train_dataset, data_fraction, fold_index, folds_number)
                self.val_dataset = self._subset(
                    self._orig_val_dataset, data_fraction, fold_index, folds_number)
                yield {'fold_index': fold_index, 'data_fraction': data_fraction}
        
        try:
            self.setup('fit')
            yield _iterate_folds()
        finally:
            for dataloader in self.val_dataloaders:
                dataloader.clear()
    
    def _subset[T: AbstractDataset](
        self,
        dataset: T,
        fraction: float,
        fold_number: int,
        folds_number: int
    ) -> T | Subset:
        if fraction is None or fraction == 1.0:
            return dataset
        
        # same pseudorandom order for every run
        rng = np.random.default_rng(seed=42)
        
        idx = rng.permutation(len(dataset.indexes))
        num = int(np.ceil(fraction * len(idx)))
        start = int(np.linspace(0, len(idx) - num, folds_number)[fold_number - 1])
        slice = idx[start:start + num]
        return Subset(dataset, slice)
    
    def _repeat[T: AbstractDataset](self, dataset: T, times: int) -> T | ConcatDataset:
        if times == 1:
            return dataset
        return ConcatDataset([dataset] * times)
    
    def get_features_info(self) -> FeaturesInfo:
        return self.train_dataset.get_features_info()
    
    def setup(self, stage: Literal['fit', 'test']):
        if stage in self.setups_completed:
            return
        
        if stage == 'fit':
            self._logger.info("Loading training data...")
            self._orig_train_dataset.load_data()
            self.scalers = self._orig_train_dataset.get_scalers()
            self._orig_val_dataset.set_scalers(self.scalers)
            self._logger.info("Loading validation data...")
            self._orig_val_dataset.load_data()
        
        if stage == 'test':
            self._logger.info("Loading test data...")
            self._orig_test_dataset.set_scalers(self.scalers)
            self._orig_test_dataset.load_data()
            
            dataset = self._repeat(
                self._orig_test_dataset, self.config.test.loader.repeat_times)
            # the test dataset has to be put in memory for consistent results
            # same for every fold
            self.test_dataset = MemDataset(dataset)
        
        self.setups_completed.add(stage)
    
    def state_dict(self):
        # To avoid saving and uploading gigabytes of scalers data,
        # we return None and rely on separate artifact
        if self.trainer is not None:
            cb = next((cb for cb in self.trainer.checkpoint_callbacks if
                       isinstance(cb, ModelCheckpoint)), None)
            if cb.save_weights_only:
                self._logger.debug("Excluding data module state from the checkpoint")
                return None
        return {'scalers': self.scalers}
    
    def load_state_dict(self, state_dict) -> None:
        if (new_scalers := state_dict.get('scalers')) is not None and self.scalers != new_scalers:
            self.scalers = new_scalers
            self._orig_train_dataset.set_scalers(new_scalers)
            self._orig_val_dataset.set_scalers(new_scalers)
            self._orig_test_dataset.set_scalers(new_scalers)
    
    def train_dataloader(self) -> DataLoader:
        if 'fit' not in self.setups_completed:
            self.setup('fit')
        
        cfg = self.config.train.loader
        dataset = self.train_dataset
        if isinstance(cfg, FinetuneDataLoaderConfig) and cfg.balanced:
            labels = np.array([i['label'][0] for i in dataset], dtype=np.int8)
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts / len(class_counts)
            sample_weights = class_weights[labels]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=max(len(dataset), cfg.batch_size),
                replacement=True,  # has to be true, otherwise it samples every element exactly once
            )
        else:
            sampler = None
        
        return DataLoader(
            dataset,
            sampler=sampler,
            drop_last=True,  # avoid model recompilation due to different batch sizes
            shuffle=sampler is None,
            collate_fn=self.collator,
            batch_size=cfg.batch_size)
    
    def val_dataloader(self) -> DataLoader:
        if 'fit' not in self.setups_completed:
            self.setup('fit')
        
        # clear the old dataloaders from GPU memory
        for dataloader in self.val_dataloaders:
            dataloader.clear()
        self.val_dataloaders.clear()
        
        cfg = self.config.val.loader
        # every time (for every fold) the new subset is returned
        dataset = self._repeat(self.val_dataset, cfg.repeat_times)
        
        # preload and sort by length
        dataset = MemDataset(dataset)
        
        # transfer to gpu
        val_dataloader = PersistentGPUDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=self.collator
        )
        self.val_dataloaders.append(val_dataloader)
        
        return val_dataloader
    
    def test_dataloader(self) -> DataLoader:
        if 'test' not in self.setups_completed:
            self.setup('test')
        
        cfg = self.config.test.loader
        
        # every time (for every fold) the same data is used
        return PersistentGPUDataLoader(
            self.test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=self.collator
        )
