import pytest

from src.util.dataset import (
    PretrainDataset,
    DatasetConfig,
    FinetuneDataset,
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
def dataset(dataset_config: DatasetConfig) -> PretrainDataset:
    return PretrainDataset(None, dataset_config)


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
    config = DatasetConfig(path='/home/user/.cache/thesis/data/preprocessed/train',
                           variables_dropout=0.2,)
    return PretrainDataset(None, config)
