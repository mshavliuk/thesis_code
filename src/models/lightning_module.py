import importlib
import logging
from dataclasses import (
    dataclass,
)
from typing import (
    Iterable,
    Sized,
)

import finetuning_scheduler as fts
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryPrecisionRecallCurve,

)
from torchmetrics.regression import MeanSquaredError
from torchmetrics.utilities.compute import auc

from src.models.strats import (
    FeaturesInfo,
    Strats,
    StratsConfig,
)
from src.util.config_namespace import ConfigNamespace


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    # trained_parameters: list[str] | Literal['all'] = 'all'


class ModuleConfig(ConfigNamespace):
    model_name: str
    model: StratsConfig
    optimizer: OptimizerConfig


class FinetuneModuleConfig(ModuleConfig):
    weighted_loss: bool


class PretrainingStrats(L.LightningModule):
    def __init__(
        self,
        module_config: ModuleConfig,
        features_info: FeaturesInfo,
    ):
        super().__init__()
        self.criterion = torch.nn.MSELoss()
        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()
        self.test_loss = MeanSquaredError()
        
        self.model = Strats(module_config.model, features_info)
        
        self.save_hyperparameters(ignore=['logger'])
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['stage'] = 'pretrain'
    
    def batch_to_inputs(self, batch):
        return {
            'values': batch['values'],
            'times': batch['times'],
            'variables': batch['variables'],
            'input_mask': batch['input_mask'],
            'demographics': batch['demographics']
        }
    
    def forward(self, values, times, variables, input_mask, demographics):
        return self.model(
            values=values,
            times=times,
            variables=variables,
            input_mask=input_mask,
            demographics=demographics
        )
    
    def _get_pred_true(self, output, batch):
        mask = batch['forecast_mask']
        pred = output[mask]
        true = batch['forecast_values'][mask]
        return pred, true
    
    def loss(self, pred, true):
        return self.criterion(pred, true)
    
    def training_step(self, batch, batch_idx):
        output = self(**self.batch_to_inputs(batch))
        pred, true = self._get_pred_true(output, batch)
        loss = self.loss(pred, true)
        self.train_loss.update(pred, true)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.log("train_epoch_loss", self.train_loss)
    
    def validation_step(self, batch, batch_idx):
        output = self(**self.batch_to_inputs(batch))
        pred, true = self._get_pred_true(output, batch)
        loss = self.loss(pred, true)
        self.val_loss.update(pred, true)
        # for some reason, by default, it is on_step=False, on_epoch=True
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        output = self(**self.batch_to_inputs(batch))
        pred, true = self._get_pred_true(output, batch)
        self.test_loss.update(pred, true)
    
    def on_test_epoch_end(self) -> None:
        self.log("test_epoch_loss", self.test_loss)
    
    def on_validation_epoch_end(self) -> None:
        self.log("val_epoch_loss", self.val_loss)
    
    def configure_optimizers(self):
        lr = self.hparams['module_config'].optimizer.lr
        return torch.optim.AdamW(self.model.parameters(), lr=lr)


class FinetuneStrats(L.LightningModule):
    def __init__(
        self,
        logger: logging.Logger,
        module_config: FinetuneModuleConfig | dict,
        features_info: FeaturesInfo,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['logger'], logger=False)
        module_config = FinetuneModuleConfig(module_config)
        self.hparams.module_config = module_config
        
        self._logger = logger
        # initialize default weight tensor to allocate memory and define initial state
        self.criterion = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(1, dtype=torch.float32))
        self.auroc = BinaryAUROC()
        self.precision_recall_curve = BinaryPrecisionRecallCurve()
        # keep different metrics separately because validation interval could be less than one epoch
        self.train_epoch_loss = MeanMetric()
        self.val_epoch_loss = MeanMetric()
        self.test_epoch_loss = MeanMetric()
        
        module_name = f"src.models.{module_config.model_name.lower()}"
        module = importlib.import_module(module_name)
        ModelClass = getattr(module, module_config.model_name.capitalize())
        self.model = ModelClass(module_config.model, features_info)
        # self.model = Strats(module_config.model, features_info)
        self.reported_logs = set()
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['stage'] = 'finetune'
    
    def on_load_checkpoint(self, checkpoint):
        state_rename_map = {  # TODO: remove this legacy code
            'model.forecast_fc.weight': 'model.head.forecast_fc.weight',
            'model.forecast_fc.bias': 'model.head.forecast_fc.bias',
            'model.binary_head.weight': 'model.head.binary_fc.weight',
            'model.binary_head.bias': 'model.head.binary_fc.bias',
        }
        
        checkpoint['state_dict'] = {
            state_rename_map.get(k, k): v for k, v in checkpoint['state_dict'].items()
        }
        
        if checkpoint['stage'] == 'pretrain':
            model_state = self.state_dict()
            for checkpoint_key in model_state:
                if checkpoint_key not in checkpoint['state_dict']:
                    checkpoint['state_dict'][checkpoint_key] = model_state[checkpoint_key]
                    self._logger.info(f"Added missing key {checkpoint_key} to checkpoint")
        
        elif checkpoint['stage'] == 'finetune':
            ...
            # self.model = Strats(self.hparams.module_config.model)
        else:
            raise ValueError("Invalid stage in checkpoint")
    
    def on_train_start(self) -> None:
        p = self._get_pos_class_frac(self.trainer.train_dataloader)
        self.wandb_logger.experiment.log({'train_pos_class_frac': p}, step=1)
        
        if self.hparams.module_config.weighted_loss:
            # avoid zero division when there are no positive samples
            # could happen during dev testing with smaller datasets
            if p == 0:
                self._logger.warning("No positive samples in training data")
                p = 0.5
            pos_class_weight = (1 - p) / p
            self._logger.info(f"Using weighted loss with weight = {pos_class_weight}")
            self.criterion.weight.fill_(pos_class_weight)
    
    def on_train_epoch_start(self) -> None:
        self.train_epoch_loss.reset()
    
    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = self.loss(output, batch)
        self.train_epoch_loss.update(loss)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_epoch_loss", self.train_epoch_loss)
    
    def on_validation_start(self) -> None:
        log_name = 'val_pos_class_frac'
        if log_name not in self.reported_logs:
            p = self._get_pos_class_frac(self.trainer.val_dataloaders)
            self.wandb_logger.experiment.log({'val_pos_class_frac': p})
            self.reported_logs.add(log_name)
    
    def on_validation_epoch_start(self) -> None:
        self.val_epoch_loss.reset()
        self.auroc.reset()
        self.precision_recall_curve.reset()
    
    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        loss = self.loss(output, batch)
        self.val_epoch_loss.update(loss)
        
        # for some reason, inside validation_step the defaults are on_step=False, on_epoch=True
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        pred = torch.sigmoid(output)
        self.auroc.update(pred, batch['label'].int())
        self.precision_recall_curve.update(pred, batch['label'].int())
        
        return loss
    
    def on_validation_epoch_end(self):
        self.log('val_epoch_loss', self.val_epoch_loss)
        self.log('val_epoch_mean_prediction', torch.cat(self.auroc.preds).mean())
        
        # auroc uses torch._cumsum which is not supported by gpu
        roc_auc = self.auroc.cpu().compute()
        self.log('val_auroc', roc_auc)
        
        # precision_recall_curve uses torch._cumsum which is not supported by gpu
        precision, recall, thresholds = self.precision_recall_curve.cpu().compute()
        
        pr_auc = auc(recall, precision)
        self.log('val_pr_auc', pr_auc)
        self.log('val_pr_roc_auc_sum', pr_auc + roc_auc)
        
        minrp = torch.min(precision, recall).max()
        self.log('val_minrp', minrp)
    
    def on_test_start(self) -> None:
        p = self._get_pos_class_frac(self.trainer.test_dataloaders)
        self.wandb_logger.experiment.log({'test_pos_class_frac': p})
    
    def on_test_epoch_start(self) -> None:
        self.test_epoch_loss.reset()
        self.auroc.reset()
        self.precision_recall_curve.reset()
    
    def test_step(self, batch, batch_idx):
        output = self(**batch)
        
        loss = self.loss(output, batch)
        self.test_epoch_loss.update(loss)
        
        pred = torch.sigmoid(output)
        self.auroc.update(pred, batch['label'].int())
        self.precision_recall_curve.update(output, batch['label'].int())
    
    def on_test_epoch_end(self):
        self.log('test_epoch_loss', self.test_epoch_loss)
        self.log('test_epoch_mean_prediction', torch.cat(self.auroc.preds).mean())
        
        auroc = self.auroc.cpu().compute()
        self.log('test_auroc', auroc)
        
        precision, recall, thresholds = self.precision_recall_curve.cpu().compute()
        
        auprc = auc(recall, precision)
        self.log('test_pr_auc', auprc)
        
        minrp = torch.min(precision, recall).max()
        self.log('test_minrp', minrp)
        self.log('test_pr_roc_auc_sum', auprc + auroc)
    
    def forward(self, values, times, variables, input_mask, demographics, **kwargs):
        return self.model(
            values=values,
            times=times,
            variables=variables,
            input_mask=input_mask,
            demographics=demographics
        )
    
    def loss(self, outputs, batch):
        return self.criterion(outputs, batch['label'])
    
    @property
    def wandb_logger(self) -> WandbLogger:
        return next(l for l in self.loggers if isinstance(l, WandbLogger))
    
    def _get_pos_class_frac(self, dataloader: DataLoader) -> float:
        if isinstance(dataloader.dataset, (Sized, Iterable)):
            return sum((i['label'][0] for i in dataloader.dataset)) / len(dataloader.dataset)
        else:
            raise ValueError("Dataset has to be sized and iterable")
    
    @property
    def finetuningscheduler_callback(self) -> fts.FinetuningScheduler:
        fts_callback = [c for c in self.trainer.callbacks if isinstance(c, fts.FinetuningScheduler)]
        return fts_callback[0] if fts_callback else None
    
    def configure_optimizers(self):
        conf: OptimizerConfig = self.hparams.module_config.optimizer
        lr = conf.lr
        if (cb := self.finetuningscheduler_callback) is not None:
            lr = cb.ft_schedule.get(0, {}).get('lr', lr)
        self._logger.info(f"Using learning rate {lr}")
        #
        # if conf.trained_parameters == 'all':
        #     trained_parameters = self.model.parameters()
        #     self._logger.info("Training all parameters")
        # else:
        #     trained_parameters = []
        #
        #     for name, param in self.model.named_parameters():
        #         if any(name.startswith(i) for i in conf.trained_parameters):
        #             trained_parameters.append(param)
        #             self._logger.info(f"Training parameter {name}")
        #         else:
        #             param.requires_grad = False
        #             self._logger.info(f"Freezing parameter {name}")
        
        # return torch.optim.AdamW(trained_parameters, lr=conf.lr)
        
        # lr is expected to be set by the scheduler
        
        return torch.optim.AdamW(self.model.parameters(), lr=lr)
