import sys
from pathlib import Path

import pytest

from src.util.dataset import (
    DatasetConfig,
    FinetuneDataset,
    FinetuneDatasetConfig,
    PretrainDataset,
    PretrainDatasetConfig,
)
from src.util.variable_scalers import (
    VariableStandardScaler,
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "scaler: provide the name of the scaler class to use for the test",
    )


@pytest.fixture(scope='module')
def data_path(request):
    path = Path('./src/util/tests/data').resolve()
    
    dir_files = {str(file.name) for file in path.glob('*')}
    
    if not path.is_dir() or \
        ({'events.parquet', 'labels.parquet', 'demographics.parquet'} - dir_files):
        print(
            '\nThe unit test data is not available.\n'
            'Run `snakemake generate_unittest_dataset` for generating the test data.',
            file=sys.stderr
        )
        pytest.skip('No unit test data available. Please provide the --data-path option.')
    
    return path


@pytest.fixture(scope='module')
def pretrain_dataset_config(data_path, request) -> PretrainDatasetConfig:
    kwargs = {
                 'path': data_path,
                 'variables_dropout': 0.2,
                 'max_events': 100,
                 'max_minute': 1000,
                 'select_top': 10,
                 'prediction_window': 10,
                 'min_input_minutes': 10
             } | request.param
    
    return PretrainDatasetConfig(**kwargs)


@pytest.fixture(scope='module')
def pretrain_dataset(pretrain_dataset_config) -> PretrainDataset:
    return PretrainDataset(pretrain_dataset_config)


@pytest.fixture(scope='module')
def finetune_dataset_config(data_path, request) -> FinetuneDatasetConfig:
    kwargs = {
                 'path': data_path,
                 'variables_dropout': 0.2,
                 'max_events': 100,
                 'max_minute': 1000,
                 'scaler_class': VariableStandardScaler.__name__,
                 'select_top': 10,
                 'prediction_window': 10,
                 'min_input_minutes': 10
             } | request.param
    
    return FinetuneDatasetConfig(**kwargs)


@pytest.fixture(scope='module')
def finetune_dataset(finetune_dataset_config: DatasetConfig) -> FinetuneDataset:
    return FinetuneDataset(finetune_dataset_config)


@pytest.fixture(scope='module')
def large_dataset() -> PretrainDataset:
    config = DatasetConfig(
        path='/home/user/.cache/thesis/data/preprocessed/train',
        variables_dropout=0.2
    )
    return PretrainDataset(None, config)
