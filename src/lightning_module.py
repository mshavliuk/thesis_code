import logging
import re
from typing import (
    Any,
    Dict,
    Iterable,
    Sized,
)

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

import src.models as models
from src.models import (
    FeaturesInfo,
    StratsConfig,
)
from src.util.config_namespace import ConfigNamespace


class ModuleConfig(ConfigNamespace):
    model_name: str
    model: StratsConfig
    optimizer: dict
    disable_bias_norm_decay: bool


class PretrainingModuleConfig(ModuleConfig):
    loss: str


class FinetuneModuleConfig(ModuleConfig):
    weighted_loss: bool


class AbstractModule(L.LightningModule):
    model: torch.nn.Module
    
    def __init__(
        self,
        module_config: dict,
        features_info: FeaturesInfo,
        logger: logging.Logger,
    ):
        super().__init__()
        self._logger = logger
    
    def build_model(self, module_config: ModuleConfig, features_info: FeaturesInfo):
        ModelClass = getattr(models, module_config.model_name)
        return ModelClass(module_config.model, features_info)
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    @property
    def wandb_logger(self) -> WandbLogger | None:
        try:
            return next(l for l in self.loggers if isinstance(l, WandbLogger))
        except StopIteration:
            return None
    
    def configure_optimizers(self):
        conf = self.hparams['module_config'].optimizer
        self._logger.info(f"Using learning rate {conf['lr']}")
        
        params = self.model.parameters()
        if self.hparams.module_config.disable_bias_norm_decay:
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
        module_config: dict,
        features_info: FeaturesInfo,
        logger: logging.Logger,
    ):
        super().__init__(module_config, features_info, logger)
        self.save_hyperparameters(ignore=['logger'], logger=False)
        module_config = PretrainingModuleConfig(module_config)
        self.hparams.module_config = module_config
        
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
        output = self(**batch)
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
        output = self(**batch)
        pred, true = self._get_pred_true(output, batch)
        loss = self.loss(pred, true)
        self.val_loss.update(pred, true)
        # for some reason, by default, it is on_step=False, on_epoch=True
        # self.log("val_loss", loss, on_step=True, on_epoch=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        output = self(**batch)
        pred, true = self._get_pred_true(output, batch)
        self.test_loss.update(pred, true)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self(**batch)
        # pred, true = self._get_pred_true(output, batch)
        return output, batch['forecast_values'], batch['forecast_mask']
    
    def on_test_epoch_end(self) -> None:
        self.log("test_epoch_loss", self.test_loss)
    
    def on_validation_epoch_end(self) -> None:
        self.log("val_epoch_loss", self.val_loss)


class FinetuneModule(AbstractModule):
    def __init__(
        self,
        module_config: FinetuneModuleConfig | dict,
        features_info: FeaturesInfo,
        logger: logging.Logger,
    ):
        super().__init__(module_config, features_info, logger)
        self.save_hyperparameters(ignore=['logger'], logger=False)
        module_config = FinetuneModuleConfig(module_config)
        self.hparams.module_config = module_config
        
        # initialize default weight tensor to allocate memory and define initial state
        self.criterion = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(1, dtype=torch.float32))
        self.auroc = BinaryAUROC()
        self.precision_recall_curve = BinaryPrecisionRecallCurve()
        # keep different metrics separately because validation interval could be less than one epoch
        # self.train_epoch_loss = MeanMetric()
        self.val_epoch_loss = MeanMetric()
        self.test_epoch_loss = MeanMetric()
        # self.train_max_input_time = MeanMetric()
        # self.train_input_sequence_length = MeanMetric()
        self.reported_logs = set()
        
        self.model = self.build_model(module_config, features_info)
    
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
    
    # def on_train_epoch_start(self) -> None:
    # self.train_epoch_loss.reset()
    
    def training_step(self, batch, batch_idx):
        # self.train_max_input_time.update(batch['times'].max(axis=1).values)
        # self.train_input_sequence_length.update(batch['input_mask'].sum(axis=1))
        output = self(**batch)
        loss = self.loss(output, batch)
        # self.train_epoch_loss.update(loss)
        # self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        ...
        # self.log("train_epoch_loss", self.train_epoch_loss)
        # self.log("train_epoch_max_input_time", self.train_max_input_time)
        # self.log("train_epoch_input_sequence_length", self.train_input_sequence_length)
    
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
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # support for legacy checkpoints without padding embedding
        state_dict = checkpoint['state_dict']
        emb = state_dict['model.variable_emb.weight']
        if emb.shape[0] == self.hparams.features_info.features_num:
            self._logger.info("Extending variable embedding with zero padding vector")
            zero_pad = torch.zeros(
                self.hparams.module_config.model.hid_dim,
                dtype=emb.dtype, device=emb.device)
            state_dict['model.variable_emb.weight'] = torch.vstack((emb, zero_pad))
    
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
        if self.wandb_logger is not None:
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
    
    def loss(self, outputs, batch):
        return self.criterion(outputs, batch['label'])
    
    def _get_pos_class_frac(self, dataloader: DataLoader) -> float:
        if isinstance(dataloader.dataset, (Sized, Iterable)):
            return sum((i['label'][0] for i in dataloader.dataset)) / len(dataloader.dataset)
        else:
            raise ValueError("Dataset has to be sized and iterable")
