import argparse
import io
import logging
from typing import Sequence

import numpy as np
import torch
import wandb

from src.lightning_module import FinetuneModule
from src.models import StratsOurs
from src.util.common import (
    create_data_module,
    create_model_module,
    create_trainer,
    create_wandb_logger,
    setup,
)
from src.util.config import (
    FinetuneConfig,
    read_config,
)
from src.util.data_module import MIMICIIIDataModule
from src.util.wandb import find_checkpoint


def finetune(
    args: argparse.Namespace, config: FinetuneConfig, logger: logging.Logger
):
    data_checkpoint, data_artifact = find_checkpoint(config, checkpoint_type='data_module')
    
    if data_checkpoint is None:
        raise FileNotFoundError("Data module checkpoint not found")
    
    data = create_data_module(config, logger, data_checkpoint)
    
    model_checkpoint, model_artifact = find_checkpoint(config, checkpoint_type='model')
    
    if model_checkpoint is None:
        raise FileNotFoundError("Model checkpoint not found")
    
    model = create_model_module(config, logger, data, model_checkpoint)
    model = model.to('cuda')
    
    values = torch.randn(16, 140, device='cuda')
    values = torch.quantize_per_tensor(values, 1 / 128, 128, torch.quint8)
    model.model.cve_value(values)
    
    values = torch.randint(0, 256, (16, 140), dtype=torch.uint8, device='cuda')
    values = torch._make_per_tensor_quantized_tensor(values, 1 / 128, 128)
    model.model.cve_value(values)
    
    # same from numpy
    values = torch.zeros(
        (16, 140),
        dtype=torch.uint8,
        pin_memory=True,
    )
    values.numpy()[:] = np.random.randint(0, 256, (16, 140), dtype=np.uint8)
    values = torch._make_per_tensor_quantized_tensor(values, 1 / 128, 128)
    model.model.cve_value(values)
    batch = next(iter(data.val_dataloader()))
    model.model.cve_value(batch['values'])
    exit()
    

    # strats: StratsOurs = model.model
    # ffn = torch.nn.Sequential(
    #     torch.ao.nn.quantized.DeQuantize(),
    #     *strats.cve_value.fnn,
    # )
    # strats.cve_value.fnn = ffn
    # strats.cve_value.forward = torch._dynamo.disable(strats.cve_value.forward)
    # modules = [new_module] + list(model)
    # model = nn.Sequential(*modules)
    # torch.quantization.prepare_qat(ffn, inplace=True)
    # ffn.eval()
    # for batch in data.train_dataloader():
    #     ffn(batch['values'].dequantize())
    #
    # ffn = torch.quantization.convert(ffn, inplace=False)
    # # lin = strats.cve_value.ffn[0]
    # # lin.weight.data /= 256
    #
    
    from torch.ao.quantization import quantize_dynamic
    # torch.quantization.quantize_dynamic(lin, dtype=torch.qint8, inplace=True)
    
    # initial_state = copy_state(model)
    # write initial state to file
    torch.save(model.state_dict(), 'initial_state.pth')
    
    # if not args.debug and not args.watch and not args.dry_run:
    #     model.compile(fullgraph=False, dynamic=True)
    
    for data_fraction in args.data_fraction:
        with data.folds(args.folds_number, data_fraction) as folds:
            for fold in folds:
                logger.info(f"Starting fold {fold}...")
                try:
                    # model.load_state_dict(initial_state, strict=True)
                    model.load_state_dict(torch.load('initial_state.pth', weights_only=False), strict=True)
                    torch.cuda.synchronize()
                    finetune_fold(args, config, fold, data, model, [model_artifact, data_artifact])
                except Exception:
                    logger.exception(f"Error in fold {fold}")


def finetune_fold(
    args: argparse.Namespace,
    config: FinetuneConfig,
    fold: dict,
    data: MIMICIIIDataModule,
    model: FinetuneModule,
    artifacts: Sequence[wandb.Artifact]
):
    wandb_logger = create_wandb_logger(args, config, artifacts, config_extra=fold)
    with wandb_logger.experiment:
        if args.watch:
            wandb_logger.watch(model, log='all')
        
        trainer = create_trainer(config, args, wandb_logger)
        
        trainer.fit(
            model=model,
            datamodule=data,
        )
        
        trainer.test(
            ckpt_path="best",
            datamodule=data,
        )
        
        if args.watch:
            wandb_logger.experiment.unwatch(model)
            model._wandb_watch_called = False  # hack to avoid wandb validation error


def copy_state(model: torch.nn.Module) -> dict:
    state_copy = {}
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor):
            if v.is_cuda:
                state_copy[k] = v.cpu()
            else:
                state_copy[k] = v.clone()
        elif isinstance(v, io.BytesIO):
            v.seek(0)
            state_copy[k] = torch.load(v)
        elif isinstance(v, (torch.dtype, tuple)):
            state_copy[k] = v
        else:
            raise ValueError(f"Unknown state_dict value type: {type(v)}")
    return state_copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to config file')
    parser.add_argument('--debug', '-d', action='store_true', help='Debug mode')
    parser.add_argument('--watch', '-w', action='store_true', help='Watch model with wandb')
    parser.add_argument('--folds-number', '-n', type=int, default=1, help='Number of folds')
    parser.add_argument('--data-fraction', '-f',
                        type=lambda line: tuple(float(v) for v in (line.split(','))),
                        default='1.0',
                        help='Fraction of data to use')
    parser.add_argument('--dry-run', action='store_true', help='Dry run')
    args = parser.parse_args()
    config = read_config(args.config)
    
    if not isinstance(config, FinetuneConfig):
        raise ValueError(f"Config stage must be 'finetune', got {config.stage}")
    
    logging.basicConfig(level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    setup()
    finetune(args, config, logger)


if __name__ == '__main__':
    main()
