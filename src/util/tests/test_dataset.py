from unittest.mock import (
    MagicMock,
    patch,
)

import numpy as np
import numpy.testing as npt
import pytest

from src.util.dataset import (
    FinetuneDataset,
)


class TestPretrainDataset:
    def test_dataset_exists(self, pretrain_dataset):
        assert pretrain_dataset is not None
    
    @pytest.fixture(scope='class')
    def loaded_dataset(self, pretrain_dataset):
        pretrain_dataset.load_data()
        return pretrain_dataset
    
    def test_dataset_getitem_returns_correct_types(self, loaded_dataset):
        item = next(iter(loaded_dataset))
        assert isinstance(item, dict)
        assert item.keys() == {
            'values', 'times', 'variables', 'demographics',
            'forecast_values', 'forecast_mask'
        }
        # check correct numpy dtypes
        assert item['values'].dtype == 'float32'
        assert item['times'].dtype == 'float32'
        assert item['variables'].dtype == 'int32'
        assert item['demographics'].dtype == 'float32'
        assert item['forecast_values'].dtype == 'float32'
        assert item['forecast_mask'].dtype == 'bool'
    
    @pytest.mark.parametrize('num_variables,p', ((5, 0.2), (10, 0.5)))
    def test_drop_variables(self, pretrain_dataset, num_variables, p):
        REPEATS = 3
        
        minutes = np.arange(0, num_variables * REPEATS)
        variables = np.repeat(np.arange(0, num_variables), REPEATS)
        values = np.arange(0, num_variables * REPEATS) * 0.1
        
        mock_rng = MagicMock()
        mock_rng.random = MagicMock(return_value=np.arange(0, 1, 1 / num_variables))
        
        with (patch.multiple(pretrain_dataset, rng=mock_rng, num_variables=num_variables),
              patch.object(pretrain_dataset.config, 'variables_dropout', p)):
            result = pretrain_dataset._drop_variables(minutes, variables, values)
        
        # Drop (p * num_variables) variables, each repeated N times in the arrays,
        # so the slice starts at (p * num_variables + 1) * N and ends at num_variables * N.
        expected_slice = slice(int((p * num_variables + 1) * REPEATS), num_variables * REPEATS)
        
        npt.assert_array_equal(result[0], minutes[expected_slice])  # minutes
        npt.assert_array_equal(result[1], variables[expected_slice])  # variables
        npt.assert_allclose(result[2], values[expected_slice]) # values
        
        mock_rng.random.assert_called_once_with(num_variables)


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
