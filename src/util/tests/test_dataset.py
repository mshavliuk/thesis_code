import tempfile

import pytest

from src.util.dataset import (
    Dataset,
    DatasetConfig,
)


class TestDataset:
    def test_dataset_exists(self, dataset):
        assert dataset is not None
    
    def test_dataset_getitem(self, dataset):
        assert isinstance(dataset[0], dict)
        assert set(dataset[0].keys()) == {'values', 'times', 'variables', 'demographics',
                                          'forecast_values', 'forecast_mask'}


class TestDatasetConfig:
    def test_dataset_config_is_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            DatasetConfig(path=tmpdir)
    
    def test_dataset_config_throws_when_path_does_not_exist(self):
        with pytest.raises(AssertionError):
            DatasetConfig(path='non_existent_dir')
