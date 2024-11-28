# Hypothesis:
#   int8 ECDF scaler fits faster and leads to faster inference as measured by number of iterations per second
# Experiment:
#   1. load ecdf model from wandb and measure test split metrics
#   2. load int8 ecdf model from wandb and measure test split metrics
#


import logging

import torch
import wandb
from torch import nn
from torch.ao.quantization import (
    DeQuantStub,
    QuantStub,
    quantize_dynamic,
)

from src.models import StratsOurs
from src.util.common import (
    create_data_module,
    create_model_module,
    create_trainer,
    setup,
)
from src.util.config import (
    FinetuneConfig,
    read_config,
)
from src.util.wandb import (
    find_checkpoint,
    get_run_checkpoint,
)


def measure_ecdf_scaler(run: wandb.apis.public.Run, config: FinetuneConfig):
    data_checkpoint, data_artifact = find_checkpoint(config, checkpoint_type='data_module')
    logger = logging.getLogger(__name__)
    
    data = create_data_module(config, logger, data_checkpoint)
    trainer = create_trainer(config)
    logger.info(f"Recomputing finetune loss summary for run {run.id}")
    checkpoint = get_run_checkpoint(run, type='model')
    model = create_model_module(config, logger, data, checkpoint)
    
    result = trainer.test(
        model=model,
        datamodule=data,
    )[0]
    
    print(result)
    
    config.data_config.test.dataset.scaler_class = 'UInt8ECDFScaler'
    data = create_data_module(config, logger, None)
    
    # test uint8 ecdf scaler
    # multiply value cve weight parameters by 1/256
    strats: StratsOurs = model.model
    strats.cve_value.fnn[0].weight.data /= 256
    result = trainer.test(
        model=model,
        datamodule=data,
    )[0]
    
    print(result)


def playground():
    values = torch.rand((16, 800), dtype=torch.float32).unsqueeze(-1)
    
    fnn = nn.Sequential(
            nn.Linear(1, 8),
            DeQuantStub(qconfig=torch.quantization.get_default_qconfig('x86')),
            nn.Tanh(),
            nn.Linear(8, 64),
        )
    emb = fnn(values)
    
    
    
    # specify quantization config for QAT
    fnn[0].qconfig = torch.quantization.get_default_qat_qconfig('x86')
    # ffn[1].qcconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # prepare QAT
    torch.quantization.prepare_qat(fnn, inplace=True)
    
    # calibrate
    # ffn(values)
    
    # convert to quantized version, removing dropout, to check for accuracy on each
    # fnn.eval()
    qfnn = torch.quantization.convert(fnn, inplace=False)
    print(qfnn)

    # scale
    values *= 256
    values = values.round().to(torch.uint8)
    # values = torch.quantize_per_tensor(values, scale=1/256, zero_point=0, dtype=torch.quint8)
    # create quantized value tensor manually
    quint8_tensor = torch._make_per_tensor_quantized_tensor(values, 1/256, 0)
    
    
    # torch.ao.quantization.convert(ffn, inplace=True)
    embq = qfnn(quint8_tensor)
    # print(embq)
    
    diffq = emb - embq.dequantize()
    
    print(diffq)


def main():
    config = read_config('experiments/finetune/ecdf.yaml')
    api = wandb.Api()
    run = api.runs(filters={
        "state": "finished",
        "display_name": config.name,
        "tags": {"$nin": ["archive", "with-bias"]},
        "config.stage": config.stage,
        'config.data_fraction': 1.0,
    }, per_page=1)[0]
    setup()
    measure_ecdf_scaler(run, config)


if __name__ == '__main__':
    # main()
    playground()
