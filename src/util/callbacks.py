import json
import os
from pathlib import Path

import lightning as L
import pandas as pd
import seaborn as sns
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from src.lightning_module import FinetuneModule
from src.util.config_namespace import MainConfig


class CurvesLoggerCallback(L.Callback):
    """
    Callback to log fold and data fraction information to the test loop.
    """
    
    def __init__(self, save_dir: str | Path):
        self.roc_fp = Path(save_dir) / 'roc_curves.jsonl'
        self.pr_fp = Path(save_dir) / 'pr_curves.jsonl'
        # self.file = open(fp, 'w')
    
    def plot_roc_curve(self, module: FinetuneModule, fpr, tpr):
        roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        fig, ax = plt.subplots()
        sns.lineplot(x='fpr', y='tpr', data=roc_data, color='blue', ax=ax)
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        self.save_fig(module, 'test_roc_curve', fig)

    
    def plot_precision_recall_curve(self, module: FinetuneModule, precision, recall):
        pr_data = pd.DataFrame({'precision': precision, 'recall': recall})
        fig, ax = plt.subplots()
        sns.lineplot(x='recall', y='precision', data=pr_data, color='blue', ax=ax)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        self.save_fig(module, 'test_pr_curve', fig)
    
    def save_fig(self, module: FinetuneModule, name: str, fig):
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
    
    def on_test_epoch_end(self, trainer: L.Trainer, module: FinetuneModule) -> None:
        roc_state = module.auroc.cpu().metric_state
        target = torch.concat(roc_state['target']).numpy()
        preds = torch.concat(roc_state['preds']).float().numpy()  # .float is needed to support bf16
        
        fpr, tpr, _ = roc_curve(target, preds)
        self.plot_roc_curve(module, fpr, tpr)
        self.save_roc_curve(fpr, tpr)
        
        precision, recall, thresholds = module.precision_recall_curve.cpu().compute()
        self.plot_precision_recall_curve(module, precision, recall)
        self.save_pr_curve(precision, recall)


def get_callbacks(config: MainConfig) -> list[L.Callback]:
    early_stop = EarlyStopping(**config.early_stop_callback)
    
    checkpoint_callback = ModelCheckpoint(
        **config.checkpoint_callback,
        filename='strats-{epoch}-{val_epoch_loss:.2f}',
        verbose=True,
    )
    
    return [early_stop, checkpoint_callback]

