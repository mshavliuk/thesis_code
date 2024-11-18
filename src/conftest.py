import sys
from pathlib import Path

import pandas as pd
import pytest

from src.util.dataset import (
    DatasetConfig,
    FinetuneDataset,
    PretrainDataset,
    PretrainDatasetConfig,
)
from src.util.variable_scalers import VariableStandardScaler


def pytest_addoption(parser):
    parser.addoption(
        "--data-path", action="store", help="my option: type1 or type2"
    )


@pytest.fixture(scope='module')
def data_path(request):
    path = Path('./src/util/tests/data')
    
    original_data_path = request.config.getoption("--data-path")
    example = '--data-path /home/user/project/data/test'
    if original_data_path is None:
        if path.exists():
            print(f'Using existing test data from {path}')
            return path
        else:
            print(
                '\nThe unit test data is not available.\n'
                'Please provide the --data-path option at lest once to generate the test data.\n'
                f'Example: pytest {example}\n'
                'Some tests will be skipped.\n', file=sys.stderr
            )
            pytest.skip('No unit test data available. Please provide the --data-path option.')
    else:
        original_data_path = Path(original_data_path)
        if not original_data_path.exists():
            print(
                f'\nThe provided data path {original_data_path} does not exist.\n'
                'Please provide a valid path to the MIMIC-III dataset.\n'
                f'Example: pytest {example}\n'
                'Some tests will be skipped.\n', file=sys.stderr
            )
            pytest.skip('The provided data path does not exist.')
        
        if path.exists():
            print(
                f'\nRemoving existing test data from {path}\n'
                'If this was not desired, avoid using --data-path option.\n', file=sys.stderr)
            for path in path.iterdir():
                path.unlink()
        
        labels = pd.read_parquet(original_data_path / 'mortality_labels.parquet').sample(frac=0.01)

        path.mkdir(parents=True, exist_ok=True)
        
        labels.to_parquet(path / 'mortality_labels.parquet')
        pd.read_parquet(
            original_data_path / 'events.parquet',
            filters=[('stay_id', 'in', labels['stay_id'])]
        ).to_parquet(path / 'events.parquet')
        pd.read_parquet(
            original_data_path / 'demographics.parquet',
            filters=[('stay_id', 'in', labels['stay_id'])]
        ).to_parquet(path / 'demographics.parquet')
        
        return path


@pytest.fixture(scope='module')
def pretrain_dataset_config(data_path) -> PretrainDatasetConfig:
    return PretrainDatasetConfig(
        path=data_path,
        variables_dropout=0.2,
        max_events=100,
        max_minute=1000,
        scaler_class=VariableStandardScaler.__name__,
        select_top=10,
        prediction_window=10,
        min_input_minutes=10,
    )


@pytest.fixture(scope='module')
def pretrain_dataset(pretrain_dataset_config) -> PretrainDataset:
    return PretrainDataset(pretrain_dataset_config)


@pytest.fixture(scope='module')
def finetune_dataset(dataset_config: DatasetConfig, dataset) -> FinetuneDataset:
    return FinetuneDataset(
        logger=None,
        config=dataset_config,
        variable_scaler=dataset.variable_scaler,
        demographic_scaler=dataset.demographic_scaler,
    )


@pytest.fixture(scope='module')
def large_dataset() -> PretrainDataset:
    config = DatasetConfig(
        path='/home/user/.cache/thesis/data/preprocessed/train',
        variables_dropout=0.2
    )
    return PretrainDataset(None, config)
