import tempfile

import pytest

from src.util.dataset import (
    FinetuneDataset,
    PretrainDataset,
    DatasetConfig,
)


class TestPretrainDataset:
    def test_dataset_exists(self, large_dataset):
        assert large_dataset is not None
    
    def test_dataset_getitem(self, dataset):
        item = dataset[0]
        assert isinstance(item, dict)
        assert set(item.keys()) == {'values', 'times', 'variables', 'demographics',
                                          'forecast_values', 'forecast_mask'}
        # check correct numpy dtypes
        assert item['values'].dtype == 'float32'
        assert item['times'].dtype == 'float32'
        assert item['variables'].dtype == 'int32'
        assert item['demographics'].dtype == 'float32'
        assert item['forecast_values'].dtype == 'float32'
        assert item['forecast_mask'].dtype == 'bool'
        


class TestFinetuneDataset:
    def test_finetune_dataset_exists(self, finetune_dataset: FinetuneDataset):
        assert finetune_dataset is not None
    
    def test_finetune_dataset_getitem(self, finetune_dataset):
        item = finetune_dataset[0]
        assert isinstance(item, dict)
        assert set(item.keys()) == {'values', 'times', 'variables', 'demographics', 'label'}
        # check correct numpy dtypes
        assert item['values'].dtype == 'float32'
        assert item['times'].dtype == 'float32'
        assert item['variables'].dtype == 'int32'
        assert item['demographics'].dtype == 'float32'
        assert item['label'].dtype == 'int32'

class TestDatasetConfig:
    def test_dataset_config_is_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            DatasetConfig(path=tmpdir)
    
    def test_dataset_config_throws_when_path_does_not_exist(self):
        with pytest.raises(AssertionError):
            DatasetConfig(path='non_existent_dir')
