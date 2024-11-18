import contextlib
import logging
import re
from typing import (
    Iterable,
    Sized,
)

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from pydantic import (
    BaseModel,
)
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryPrecisionRecallCurve,

)
from torchmetrics.regression import MeanSquaredError
from torchmetrics.utilities.compute import auc

import src.models as models
from src.models import (
    FeaturesInfo,
    StratsConfig,
)


class ModuleConfig(BaseModel, extra='forbid'):
    class_name: str
    model: StratsConfig
    optimizer: dict
    disable_bias_norm_decay: bool


class PretrainingModuleConfig(ModuleConfig, extra='forbid'):
    loss: str


class FinetuneModuleConfig(ModuleConfig, extra='forbid'):
    weighted_loss: bool


class AbstractModule(L.LightningModule):
    model: torch.nn.Module
    
    def __init__(
        self,
        module_config: ModuleConfig,
        features_info: FeaturesInfo,
        logger: logging.Logger,
    ):
        super().__init__()
        self._logger = logger
        self.config = module_config
        self.features_info = features_info
    
    def build_model(self, module_config: ModuleConfig, features_info: FeaturesInfo):
        ModelClass = getattr(models, module_config.class_name)
        return ModelClass(module_config.model, features_info)
    
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(
            values=batch['values'],
            times=batch['times'],
            variables=batch['variables'],
            input_mask=batch['input_mask'],
            demographics=batch['demographics'],
        )
    
    @property
    def wandb_logger(self) -> WandbLogger | None:
        try:
            return next(l for l in self.loggers if isinstance(l, WandbLogger))
        except StopIteration:
            return None
    
    def configure_optimizers(self):
        conf = self.config.optimizer
        self._logger.info(f"Using learning rate {conf['lr']}")
        
        params = self.model.parameters()
        if self.config.disable_bias_norm_decay:
            self._logger.info("Disabling bias and norm decay")
            
            bias_re = re.compile(r'\.b\d?$')
            
            def should_decay(layer_name: str) -> bool:
                return not (
                    'bias' in layer_name or 'norm' in layer_name or bias_re.search(layer_name))
            
            params = [
                {
                    'params': [p for n, p in self.model.named_parameters() if not should_decay(n)],
                    'weight_decay': 0.0,
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if should_decay(n)],
                },
            ]
        
        optimizer = torch.optim.AdamW(params, **conf)
        return optimizer


class PretrainingModule(AbstractModule):
    def __init__(
        self,
        module_config: PretrainingModuleConfig,
        features_info: FeaturesInfo,
        logger: logging.Logger,
    ):
        super().__init__(module_config, features_info, logger)
        self.save_hyperparameters(ignore=['logger'], logger=False)
        self.criterion = getattr(torch.nn, module_config.loss)()
        self.val_loss = MeanSquaredError()
        self.test_loss = MeanSquaredError()
        # self.train_max_input_time = MeanMetric()
        # self.train_input_sequence_length = MeanMetric()
        # self.train_pred_features_number = MeanMetric()
        self.model = self.build_model(module_config, features_info)
    
    def _get_pred_true(self, output, batch):
        mask = batch['forecast_mask']
        pred = output[mask]
        true = batch['forecast_values'][mask]
        return pred, true
    
    def loss(self, pred, true):
        return self.criterion(pred, true)
    
    def training_step(self, batch, batch_idx):
        # self.train_max_input_time.update(batch['times'].max(axis=1).values)
        # self.train_input_sequence_length.update(batch['input_mask'].sum(axis=1))
        # self.train_pred_features_number.update(batch['forecast_mask'].sum(axis=1))
        output = self(batch)
        pred, true = self._get_pred_true(output, batch)
        loss = self.loss(pred, true)
        # self.train_loss.update(pred, true)
        # self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss
    
    # def on_train_epoch_end(self) -> None:
    #     self.log("train_epoch_loss", self.train_loss)
    #     self.log("train_epoch_max_input_time", self.train_max_input_time)
    #     self.log("train_epoch_input_sequence_length", self.train_input_sequence_length)
    #     self.log("train_epoch_pred_features_number", self.train_pred_features_number)
    
    def validation_step(self, batch, batch_idx):
        output = self(batch)
        pred, true = self._get_pred_true(output, batch)
        loss = self.loss(pred, true)
        self.val_loss.update(pred, true)
        # for some reason, by default, it is on_step=False, on_epoch=True
        # self.log("val_loss", loss, on_step=True, on_epoch=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        output = self(batch)
        pred, true = self._get_pred_true(output, batch)
        self.test_loss.update(pred, true)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self(batch)
        # pred, true = self._get_pred_true(output, batch)
        return output, batch['forecast_values'], batch['forecast_mask']
    
    def on_test_epoch_end(self) -> None:
        self.log("test_epoch_loss", self.test_loss)
    
    def on_validation_epoch_end(self) -> None:
        self.log("val_epoch_loss", self.val_loss)


class FinetuneModule(AbstractModule):
    def __init__(
        self,
        module_config: FinetuneModuleConfig,
        features_info: FeaturesInfo,
        logger: logging.Logger,
    ):
        super().__init__(module_config, features_info, logger)
        self.save_hyperparameters(ignore=['logger'], logger=False)
        # initialize default weight tensor to allocate memory and define initial state
        self.criterion = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(1, dtype=torch.float32))
        self.auroc = BinaryAUROC()
        self.precision_recall_curve = BinaryPrecisionRecallCurve()
        # keep different metrics separately because validation interval could be less than one epoch
        # self.train_epoch_loss = MeanMetric()
        self.val_epoch_loss = MeanMetric()
        self.test_epoch_loss = MeanMetric()
        self.reported_logs = set()
        
        self.model = self.build_model(module_config, features_info)
    
    def on_train_start(self) -> None:
        if self.wandb_logger is not None:
            p = self._get_pos_class_frac(self.trainer.train_dataloader)
            self.wandb_logger.experiment.log({'train_pos_class_frac': p}, step=1)
        
        if self.config.weighted_loss:
            pos_class_weight = self._get_pos_class_weight(self.trainer.train_dataloader)
            self._logger.info(f"Using weighted BCE loss with weight = {pos_class_weight}")
            self.criterion.weight.fill_(pos_class_weight)
        else:
            self._logger.info("Using unweighted BCE loss")
    
    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(output, batch)
        return loss
    
    def on_validation_start(self) -> None:
        log_name = 'val_pos_class_frac'
        if log_name not in self.reported_logs and self.wandb_logger is not None:
            p = self._get_pos_class_frac(self.trainer.val_dataloaders)
            self.wandb_logger.experiment.log({log_name: p})
            self.reported_logs.add(log_name)
    
    def on_validation_epoch_start(self) -> None:
        self._reset_metrics()
    
    def validation_step(self, batch, batch_idx):
        output = self(batch)
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
        metrics = self._compute_metrics()
        self.log_dict({f'val_{k}': v for k, v in metrics.items()})
    
    def on_test_start(self) -> None:
        if self.wandb_logger is not None:
            p = self._get_pos_class_frac(self.trainer.test_dataloaders)
            self.wandb_logger.experiment.log({'test_pos_class_frac': p})
    
    def on_test_epoch_start(self) -> None:
        self._reset_metrics()
    
    def test_step(self, batch, batch_idx):
        output = self(batch)
        
        loss = self.loss(output, batch)
        self.test_epoch_loss.update(loss)
        
        pred = torch.sigmoid(output)
        self.auroc.update(pred, batch['label'].int())
        self.precision_recall_curve.update(output, batch['label'].int())
    
    def on_test_epoch_end(self):
        self.log('test_loss', self.test_epoch_loss)
        
        metrics = self._compute_metrics()
        self.log_dict({f'test_{k}': v for k, v in metrics.items()})
    
    def _compute_metrics(self):
        # auroc uses torch._cumsum which is not supported by gpu
        roc_auc = self.auroc.cpu().compute()
        
        # precision_recall_curve uses torch._cumsum which is not supported by gpu
        precision, recall, thresholds = self.precision_recall_curve.cpu().compute()
        
        pr_auc = auc(recall, precision)
        
        minrp = torch.min(precision, recall).max()
        return {
            'mean_prediction': torch.cat(self.auroc.preds).mean(),
            'auroc': roc_auc,
            'pr_auc': pr_auc,
            'pr_roc_auc_sum': pr_auc + roc_auc,
            'minrp': minrp,
        }
    
    def _reset_metrics(self):
        self.auroc.reset()
        self.precision_recall_curve.reset()
        self.val_epoch_loss.reset()
        self.test_epoch_loss.reset()
    
    def loss(self, outputs, batch):
        return self.criterion(outputs, batch['label'])
    
    @contextlib.contextmanager
    def with_weighted_loss(self, train_dataloader: DataLoader):
        if self.criterion.weight is not None and self.criterion.weight != 1:
            yield  # already weighted
        else:
            previous_criterion = self.criterion
            try:
                pos_class_weight = self._get_pos_class_weight(train_dataloader)
                self._logger.info(f"Using weighted BCE loss with weight = {pos_class_weight}")
                self.criterion = torch.nn.BCEWithLogitsLoss(
                    weight=torch.tensor(pos_class_weight, dtype=torch.float32))
                yield
            finally:
                self.criterion = previous_criterion
    
    @contextlib.contextmanager
    def with_unweighted_loss(self):
        if self.criterion.weight is None or self.criterion.weight == 1:
            yield # already unweighted
        else:
            previous_criterion = self.criterion
            try:
                self._logger.info("Using unweighted BCE loss")
                self.criterion = torch.nn.BCEWithLogitsLoss()
                yield
            finally:
                self.criterion = previous_criterion
    
    def _get_pos_class_frac(self, dataloader: DataLoader) -> float:
        if isinstance(dataloader.dataset, (Sized, Iterable)):
            return sum((i['label'][0] for i in dataloader.dataset)) / len(dataloader.dataset)
        else:
            raise ValueError("Dataset has to be sized and iterable")
    
    def _get_pos_class_weight(self, dataloader: DataLoader) -> float:
        p = self._get_pos_class_frac(dataloader)
        if p == 0:
            # avoid zero division when there are no positive samples
            # could happen during dev runs with small datasets
            self._logger.warning("No positive samples in training data")
            return 1.0
        
        pos_class_weight = (1 - p) / p
        return pos_class_weight
