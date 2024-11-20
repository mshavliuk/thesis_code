from unittest.mock import (
    MagicMock,
    patch,
)

import numpy as np
import numpy.testing as npt
import pytest

import src.util.variable_scalers as scalers_module
from src.models import FeaturesInfo
from src.util.dataset import (
    FinetuneDataset,
    FinetuneDatasetConfig,
    PretrainDataset,
    PretrainDatasetConfig,
)
from src.util.variable_scalers import (
    DemographicScaler,
    TimeScaler,
    VariableECDFScaler,
    VariableStandardScaler,
)


@pytest.fixture(scope='class')
def loaded_pretrain_dataset(pretrain_dataset) -> PretrainDataset:
    pretrain_dataset.load_data()
    return pretrain_dataset


@pytest.fixture(scope='class')
def loaded_finetune_dataset(
    finetune_dataset,
    loaded_pretrain_dataset,
) -> FinetuneDataset:
    scalers = loaded_pretrain_dataset.get_scalers()
    finetune_dataset.set_scalers(scalers)
    finetune_dataset.load_data()
    return finetune_dataset


@pytest.mark.parametrize('pretrain_dataset_config', [
    {'scaler_class': VariableECDFScaler.__name__},
    {'scaler_class': VariableStandardScaler.__name__}
], indirect=['pretrain_dataset_config'])
class TestPretrainDataset:
    def test_get_features_info(self, pretrain_dataset):
        features_info = pretrain_dataset.get_features_info()
        assert isinstance(features_info, FeaturesInfo)
        assert features_info.demographics_num == 2
    
    def test_dataset_getitem_throws_error(self, pretrain_dataset):
        with pytest.raises(ValueError, match='Data has not been loaded'):
            next(iter(pretrain_dataset))
    
    def test_dataset_getitem_returns_correct_types(self, loaded_pretrain_dataset):
        item = next(iter(loaded_pretrain_dataset))
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
    
    def test_get_scalers(
        self,
        loaded_pretrain_dataset,
        pretrain_dataset_config: PretrainDatasetConfig
    ):
        scalers = loaded_pretrain_dataset.get_scalers()
        assert isinstance(scalers, dict)
        assert set(scalers.keys()) == {'time_scaler', 'demographic_scaler', 'variable_scaler'}
        
        assert isinstance(scalers['time_scaler'], TimeScaler)
        assert scalers['time_scaler'].half_max_minute == pretrain_dataset_config.max_minute / 2
        
        assert isinstance(scalers['demographic_scaler'], DemographicScaler)
        assert scalers['demographic_scaler'].age_mean != 0
        assert scalers['demographic_scaler'].age_std != 0
        
        assert isinstance(scalers['variable_scaler'],
                          getattr(scalers_module, pretrain_dataset_config.scaler_class))
    
    @pytest.mark.parametrize('num_variables,p', ((5, 0.2), (10, 0.5)))
    def test_drop_variables(self, loaded_pretrain_dataset, num_variables, p):
        REPEATS = 3
        
        minutes = np.arange(0, num_variables * REPEATS)
        variables = np.repeat(np.arange(0, num_variables), REPEATS)
        values = np.arange(0, num_variables * REPEATS) * 0.1
        
        mock_rng = MagicMock()
        mock_rng.random = MagicMock(return_value=np.arange(0, 1, 1 / num_variables))
        
        with (patch.multiple(loaded_pretrain_dataset, rng=mock_rng, num_variables=num_variables),
              patch.object(loaded_pretrain_dataset.config, 'variables_dropout', p)):
            result = loaded_pretrain_dataset._drop_variables(minutes, variables, values)
        
        # Drop (p * num_variables) variables, each repeated N times in the arrays,
        # so the slice starts at (p * num_variables + 1) * N and ends at num_variables * N.
        expected_slice = slice(int((p * num_variables + 1) * REPEATS), num_variables * REPEATS)
        
        npt.assert_array_equal(result[0], minutes[expected_slice])  # minutes
        npt.assert_array_equal(result[1], variables[expected_slice])  # variables
        npt.assert_allclose(result[2], values[expected_slice])  # values
        
        mock_rng.random.assert_called_once_with(num_variables)


@pytest.mark.parametrize('finetune_dataset_config,pretrain_dataset_config', [
    [{'scaler_class': VariableECDFScaler.__name__}] * 2,
    [{'scaler_class': VariableStandardScaler.__name__}] * 2
], indirect=['finetune_dataset_config', 'pretrain_dataset_config'])
class TestFinetuneDataset:
    def test_finetune_dataset_getitem_throws_error(
        self,
        finetune_dataset: FinetuneDataset,
        pretrain_dataset_config,
    ):
        with pytest.raises(ValueError, match='Data has not been loaded'):
            next(iter(finetune_dataset))
    
    def test_get_scalers(
        self,
        loaded_finetune_dataset,
        finetune_dataset_config: FinetuneDatasetConfig
    ):
        scalers = loaded_finetune_dataset.get_scalers()
        assert isinstance(scalers, dict)
        assert set(scalers.keys()) == {'time_scaler', 'demographic_scaler', 'variable_scaler'}
        
        assert isinstance(scalers['time_scaler'], TimeScaler)
        assert scalers['time_scaler'].half_max_minute == loaded_finetune_dataset.max_minute / 2
        
        assert isinstance(scalers['demographic_scaler'], DemographicScaler)
        assert scalers['demographic_scaler'].age_mean != 0
        assert scalers['demographic_scaler'].age_std != 0
        
        assert isinstance(scalers['variable_scaler'],
                          getattr(scalers_module, finetune_dataset_config.scaler_class))
    
    def test_finetune_dataset_getitem(
        self,
        loaded_finetune_dataset: FinetuneDataset,
    ):
        item = next(iter(loaded_finetune_dataset))
        assert isinstance(item, dict)
        assert set(item.keys()) == {'values', 'times', 'variables', 'demographics', 'label'}
        # check correct numpy dtypes
        assert item['values'].dtype == 'float32'
        assert item['times'].dtype == 'float32'
        assert item['variables'].dtype == 'int32'
        assert item['demographics'].dtype == 'float32'
        assert item['label'].dtype == 'float32'
