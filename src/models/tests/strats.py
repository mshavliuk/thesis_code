import time

import pytest
import torch.cuda

from src.models.strats import (
    Strats,
    StratsConfig,
)
from src.util.collator import Collator
from src.util.data_loader import (
    Batch,
    DataLoader,
)
from src.models.losses import MaskedMSELoss

HEAD_CONFIGS = {
    'forecast': {'head': 'forecast'},
    'binary': {'head': 'binary'},
    'forecast_binary': {'head': 'forecast_binary'},
}


class TestStrats:
    @pytest.fixture()
    def config(self, dataset, request):
        defaults = dict(
            demographics_num=2,  # number of demographics features, computed from data
            features_num=dataset.num_variables,  # number of variables, computed from data
            hid_dim=64,
            num_layers=2,
            num_heads=16,
            dropout=0.2,
            attention_dropout=0.2,
            head='binary',
        )
        if hasattr(request, 'param'):
            return StratsConfig(**{**defaults, **request.param})
        return StratsConfig(**defaults)
    
    @pytest.fixture()
    def device(self, request):
        if hasattr(request, 'param'):
            return request.param
        return 'cpu'
    
    @pytest.fixture()
    def model(self, config, device):
        model = Strats(config)
        return model.to(device)
    
    @pytest.fixture(scope='function')
    def data_loader(self, request, dataset, device):
        data_loader = DataLoader(dataset, device=device, batch_size=4, collate_fn=Collator())
        return data_loader
    
    @pytest.fixture()
    def batch(self, data_loader):
        return next(iter(data_loader))
    
    @pytest.mark.parametrize('config', HEAD_CONFIGS.values(), indirect=True, ids=HEAD_CONFIGS.keys())
    def test_init(self, model, config):
        assert model.args == config
        assert model.dropout == config.dropout
        assert model.features_num == config.features_num
    
    @pytest.mark.parametrize('config', HEAD_CONFIGS.values(), indirect=True, ids=HEAD_CONFIGS.keys())
    @pytest.mark.parametrize('device', ['cpu', 'cuda'], indirect=True)
    def test_forward_backward(self, model: Strats, batch: Batch):
        output = model(**batch.model_inputs)
        assert output is not None
        # todo: change loss for binary head
        loss = MaskedMSELoss()(output, batch['forecast_values'], batch['forecast_mask'])
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())
        
    
    # @pytest.mark.parametrize('config', HEAD_CONFIGS.values(), indirect=True, ids=HEAD_CONFIGS.keys())
    @pytest.mark.parametrize('device', ['cpu', 'cuda'], indirect=True)
    def test_compile(self, model: Strats, batch: Batch, device):
        # TODO: torch.jit.script(model)
        
        model = model.to(device)
        model = torch.compile(model)
        output = model(**batch.model_inputs)
        # true =
        forecast_mask = batch['forecast_mask'].to(device)
        forecast_values = batch['forecast_values'].to(device)
        loss = (forecast_mask * (output - forecast_values) ** 2).sum() / forecast_mask.sum()
        loss.backward()
    
    @pytest.mark.parametrize('config', HEAD_CONFIGS.values(), indirect=True, ids=HEAD_CONFIGS.keys())
    def test_benchmark_forward(self, model: Strats, data_loader):
        start_time = time.time()
        device = 'cpu'
        model.to(device)
        for _, batch in zip(range(1000), data_loader):
            output = model(**batch.model_inputs)
            torch.cuda.synchronize()
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Time per batch: {total_time / 1000:.4f} seconds {1000 / total_time}(it/sec)")
