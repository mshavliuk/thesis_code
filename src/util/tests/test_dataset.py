import tempfile
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest

from src.util.dataset import (
    DatasetConfig,
    FinetuneDataset,
    PretrainDataset,
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
    
    def test_drop_variables(self, dataset: PretrainDataset):
        variables = np.repeat(np.arange(1, 6), 2)
        minutes = np.arange(0, 10)
        values = np.arange(0, 10) * 0.1
        with patch.object(dataset, 'num_variables', 5):
            np.random.seed(1)
            result = dataset._drop_variables(minutes, variables, values)
        npt.assert_array_equal(result[0], np.array([0, 1, 2, 3, 6, 7]))  # minutes
        npt.assert_array_equal(result[1], np.array([1, 1, 2, 2, 4, 4]))  # variables
        npt.assert_allclose(result[2], np.array([0, 0.1, 0.2, 0.3, 0.6, 0.7])) # values
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
