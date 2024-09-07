import json
import os
from pathlib import Path

import lightning as L
import pandas as pd
import seaborn as sns
import torch
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from src.models.lightning_module import FinetuneStrats


class CurvesLoggerCallback(L.Callback):
    """
    Callback to log fold and data fraction information to the test loop.
    """
    
    def __init__(self, save_dir: str | Path):
        self.roc_fp = Path(save_dir) / 'roc_curves.jsonl'
        self.pr_fp = Path(save_dir) / 'pr_curves.jsonl'
        # self.file = open(fp, 'w')
    
    def plot_roc_curve(self, module: FinetuneStrats, fpr, tpr):
        roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        fig, ax = plt.subplots()
        sns.lineplot(x='fpr', y='tpr', data=roc_data, color='blue', ax=ax)
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        self.save_fig(module, 'test_roc_curve', fig)

    
    def plot_precision_recall_curve(self, module: FinetuneStrats, precision, recall):
        pr_data = pd.DataFrame({'precision': precision, 'recall': recall})
        fig, ax = plt.subplots()
        sns.lineplot(x='recall', y='precision', data=pr_data, color='blue', ax=ax)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        self.save_fig(module, 'test_pr_curve', fig)
    
    def save_fig(self, module: FinetuneStrats, name: str, fig):
        wandb_logger = next(l for l in module.loggers if isinstance(l, WandbLogger))
        
        fig.tight_layout()
        fig.set_dpi(300)
        
        wandb_logger.log_image(name, [fig])
        plt.close(fig)
    
    def save_roc_curve(self, fpr, tpr):
        with self.roc_fp.open('a') as file:
            file.write(json.dumps({
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
            }) + '\n')
    
    def save_pr_curve(self, precision, recall):
        with self.pr_fp.open('a') as file:
            file.write(json.dumps({
                'precision': precision.tolist(),
                'recall': recall.tolist(),
            }) + '\n')
    
    def on_test_epoch_end(self, trainer: L.Trainer, module: FinetuneStrats) -> None:
        roc_state = module.auroc.cpu().metric_state
        target = torch.concat(roc_state['target']).numpy()
        preds = torch.concat(roc_state['preds']).float().numpy()  # .float is needed to support bf16
        
        fpr, tpr, _ = roc_curve(target, preds)
        self.plot_roc_curve(module, fpr, tpr)
        self.save_roc_curve(fpr, tpr)
        
        precision, recall, thresholds = module.precision_recall_curve.cpu().compute()
        self.plot_precision_recall_curve(module, precision, recall)
        self.save_pr_curve(precision, recall)


import wandb
from lightning.pytorch.callbacks import ModelCheckpoint


class WandbModelCheckpoint(ModelCheckpoint):
    # TODO: delete this class
    def __init__(self, *args, wandb_logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb_logger = wandb_logger
        self.best_model_artifact: wandb.Artifact | None = None
    
    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        
        # upload if has to store every epoch
        if self.save_top_k == -1:
            artifact = self._upload_artifact(filepath)
            
            if self.current_score == self.best_model_score:
                self.best_model_artifact = artifact
    
    def _upload_artifact(self, filepath, **kwargs) -> wandb.Artifact:
        model_checkpoint_artifact = wandb.Artifact(
            name=self.wandb_logger.experiment.name,
            type='checkpoint',
            metadata={
                'score_value': self.current_score,
                'score_metric': self.monitor,
                'filepath': filepath,
            }
        )
        if os.path.isfile(filepath):
            model_checkpoint_artifact.add_file(filepath)
        elif os.path.isdir(filepath):
            model_checkpoint_artifact.add_dir(filepath)
        else:
            raise FileNotFoundError(f"No such file or directory {filepath}")
        
        return self.wandb_logger.experiment.log_artifact(model_checkpoint_artifact, **kwargs)
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.best_model_artifact:
            self.best_model_artifact.wait()
            self.best_model_artifact.aliases += ['best']
            self.best_model_artifact.save()
        else:
            best_checkpoint=  self.best_model_path
            self._upload_artifact(best_checkpoint, aliases=['best'])
