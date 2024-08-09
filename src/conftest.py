import pytest

from src.util.dataset import (
    Dataset,
    DatasetConfig,
)


@pytest.fixture(scope='module')
def dataset_config():
    # with tempfile.TemporaryDirectory() as tmpdir:
    CONFIG_SAMPLE = {
        'some_config': {},
        'dataset': {
            'path': './src/util/tests/data',
            'variables_dropout': 0.2,
        }
    }
    
    yield DatasetConfig(**CONFIG_SAMPLE['dataset'])


@pytest.fixture(scope='module')
def dataset(dataset_config: DatasetConfig) -> Dataset:
    return Dataset(logger=None, config=dataset_config)


@pytest.fixture(scope='module')
def large_dataset() -> Dataset:
    config = DatasetConfig(path='/home/user/.cache/thesis/data/preprocessed/train')
    return Dataset(logger=None, config=config)
