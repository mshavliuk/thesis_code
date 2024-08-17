import logging
from argparse import Namespace
from dataclasses import (
    asdict,
    dataclass,
)

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryPrecisionRecallCurve,

)
from torchmetrics.regression import MeanSquaredError
from torchmetrics.utilities.compute import auc

from src.models.strats import (
    Strats,
    StratsConfig,
)


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float


class ModuleConfig(Namespace):
    model: StratsConfig
    optimizer: OptimizerConfig
    
    def __init__(self, **kwargs):
        super().__init__()
        for prop in self.__annotations__:
            field_type = self.__annotations__[prop]
            if isinstance(kwargs[prop], field_type):
                setattr(self, prop, kwargs[prop])
            else:
                setattr(self, prop, field_type(**kwargs[prop]))


class PretrainingStrats(L.LightningModule):
    def __init__(self,
        module_config: ModuleConfig,
        learning_rate: float = 1e-5,
    ):
        super().__init__()
        self.criterion = torch.nn.MSELoss()
        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()
        self.test_loss = MeanSquaredError()
        
        self.model = Strats(module_config.model)
        
        self.save_hyperparameters(ignore=['logger'])
    
    def on_save_checkpoint(self, checkpoint):
        # checkpoint['scalers'] = self.trainer.train_dataloader.dataset.get_scalers()
        checkpoint['stage'] = 'pretrain'
    
    # def on_load_checkpoint(self, checkpoint):
    #     for key, val in self.hparams.items():
    #         if checkpoint['hyper_parameters'][key] != val:
    #             raise ValueError(f"Hparams mismatch {checkpoint['hyper_parameters']} != {self.hparams}")

    
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
        lr = self.hparams['learning_rate'] or self.hparams['module_config'].optimizer.lr
        print(f"Learning rate: {lr}")
        return torch.optim.AdamW(self.model.parameters(), lr=lr)


class FinetuneStrats(L.LightningModule):
    def __init__(self, logger: logging.Logger, optimizer: OptimizerConfig):
        super().__init__()
        self._logger = logger
        self.pos_class_weight = None
        # initialize default weight tensor to allocate memory and define initial state
        self.criterion = torch.nn.BCELoss(weight=torch.tensor(1, dtype=torch.float32))
        self.auroc = BinaryAUROC()
        self.precision_recall_curve = BinaryPrecisionRecallCurve()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
        self.model = None
        self.scalers = None
        self.save_hyperparameters(ignore=['logger'])
    
    def on_save_checkpoint(self, checkpoint):
        # can't access self.trainer.train_dataloader.dataset.get_scalers() because a Subset is given
        # checkpoint['scalers'] = self.scalers
        checkpoint['stage'] = 'finetune'
    
    def on_load_checkpoint(self, checkpoint):
        # store just to save to new checkpoint later
        # self.scalers = checkpoint['scalers']
        
        hparams = checkpoint['hyper_parameters']
        stats_config = asdict(hparams['model'])
        
        if checkpoint['stage'] == 'pretrain':
            stats_config['head'] = 'forecast_binary'
            finetune_strats_config = StratsConfig(**stats_config)
            
            self.model = Strats(finetune_strats_config)
            model_state = self.state_dict()
            for checkpoint_key in model_state:
                if checkpoint_key not in checkpoint['state_dict']:
                    checkpoint['state_dict'][checkpoint_key] = model_state[checkpoint_key]
                    self._logger.info(f"Added missing key {checkpoint_key} to checkpoint")
        
        elif checkpoint['stage'] == 'finetune':
            finetune_strats_config = StratsConfig(**stats_config)
            self.model = Strats(finetune_strats_config)
        else:
            raise ValueError("Invalid stage in checkpoint")
        
        hparams = ModuleConfig(
            model=finetune_strats_config,
            optimizer=self.hparams['optimizer'],
            # TODO: make sure optimizer config is not redefined by checkpoint
        )
        self.save_hyperparameters(hparams)
    
    def training_step(self, batch, batch_idx):
        output = self(**self.batch_to_inputs(batch))
        loss = self.loss(output, batch)
        self.train_loss.update(loss)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_epoch_loss", self.train_loss)
    
    def batch_to_inputs(self, batch):
        return {
            'values': batch['values'],
            'times': batch['times'],
            'variables': batch['variables'],
            'input_mask': batch['input_mask'],
            'demographics': batch['demographics']
        }
    
    def validation_step(self, batch, batch_idx):
        output = self(**self.batch_to_inputs(batch))
        loss = self.loss(output, batch)
        self.val_loss.update(loss)
        
        # FIXME: remove
        # self.trainer._results.update({'fold': _ResultMetric(1, False)})
        
        self.log("val_loss", loss, prog_bar=True)
        self.auroc.update(output, batch['label'].int())
        self.precision_recall_curve.update(output, batch['label'].int())
        
        return loss
    
    def on_validation_epoch_end(self):
        self.log('val_epoch_loss', self.val_loss)
        
        # auroc uses torch._cumsum which is not supported by gpu
        roc_auc = self.auroc.cpu().compute()
        self.log('val_auroc', roc_auc)
        self.auroc.reset()
        
        # precision_recall_curve uses torch._cumsum which is not supported by gpu
        precision, recall, thresholds = self.precision_recall_curve.cpu().compute()
        self.precision_recall_curve.reset()
        
        pr_auc = auc(recall, precision)
        self.log('val_pr_auc', pr_auc)
        self.log('val_pr_roc_auc_sum', pr_auc + roc_auc)
        
        minrp = torch.min(precision, recall).max()
        self.log('val_minrp', minrp)
    
    def test_step(self, batch, batch_idx):
        output = self(**self.batch_to_inputs(batch))
        self.auroc.update(output, batch['label'].int())
        self.precision_recall_curve.update(output, batch['label'].int())
    
    def on_test_epoch_end(self):
        roc_auc = self.auroc.cpu().compute()
        self.log('test_auroc', roc_auc)
        
        precision, recall, thresholds = self.precision_recall_curve.cpu().compute()
        
        pr_auc = auc(recall, precision)
        self.log('test_pr_auc', pr_auc)
        
        minrp = torch.min(precision, recall).max()
        self.log('test_minrp', minrp)
        self.log('test_pr_roc_auc_sum', pr_auc + roc_auc)
    
    def forward(self, values, times, variables, input_mask, demographics):
        return self.model(
            values=values,
            times=times,
            variables=variables,
            input_mask=input_mask,
            demographics=demographics
        )
    
    def loss(self, outputs, batch):
        return self.criterion(outputs, batch['label'])
    
    def on_train_start(self):
        if self.model is None:
            raise ValueError("Model has to be loaded from checkpoint")
        
        pos_class_frac = MeanMetric()
        for batch in self.trainer.train_dataloader:
            pos_class_frac.update(batch['label'])
        p = pos_class_frac.compute()
        pos_class_weight = (1 - p) / p
        self.criterion.weight = pos_class_weight.to(self.device)
        wandb_logger = next(l for l in self.loggers if isinstance(l, WandbLogger))
        wandb_logger.experiment.log({'pos_class_frac': p})
        # if isinstance(run := getattr(self.logger, 'experiment', None), WandbRun):
        #     run.log({'pos_class_frac': p})
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams['optimizer'].lr)
