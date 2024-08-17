import inspect
import json
import re
from pathlib import Path

import lightning as L
import pandas as pd
import seaborn as sns
import torch
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from src.models.lightning_module import FinetuneStrats


class FoldInfoLoggerCallback(L.Callback):
    """
    Callback to log fold and data fraction information to the test loop.
    """
    
    def __init__(self, fold: int, data_fraction: float):
        self.fold = fold
        self.data_fraction = data_fraction
    
    def log_fold_info(self, trainer: L.Trainer, module: L.LightningModule):
        module.log_dict({'fold': self.fold, 'data_fraction': self.data_fraction})
    
    def setup(self, trainer: L.Trainer, module: L.LightningModule, stage: str) -> None:
        overridden_hooks = []
        module_methods = inspect.getmembers(type(module), predicate=inspect.isfunction)
        hook_re = re.compile(r'^on_(\w+)_(start|end|epoch_start|epoch_end|step)$')
        
        for name, method in module_methods:
            if hook_re.match(name):
                if parent_method := getattr(L.LightningModule, name, None):
                    if method.__code__ != parent_method.__code__:
                        overridden_hooks.append(name)
        
        for hook_name in overridden_hooks:
            setattr(self, hook_name, self.log_fold_info)


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
        # TRY to use wandb table
        #
        # wandb_logger.log_table('test_roc_curve_table', columns=['fpr', 'tpr'], data=list(zip(fpr, tpr)))
        # import wandb
        # table = wandb.Table(data=list(zip(fpr, tpr)), columns=['fpr', 'tpr'])
        # # save plot
        # # wandb.log({"roc_curve": wandb.plot.roc_curve(target, preds)})
        # plot = wandb_logger.experiment.plot_table('test_roc_curve_plot', table, {"x": "fpr", "y": "tpr"},
        #                  {
        #                         "title": "ROC Curve",
        #                         "x-axis-title": "False positive rate",
        #                         "y-axis-title": "True positive rate",
        #                  })
        # wandb_logger.experiment.log({"roc_curve": plot})
    
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
        preds = torch.concat(roc_state['preds']).numpy()
        
        fpr, tpr, _ = roc_curve(target, preds)
        self.plot_roc_curve(module, fpr, tpr)
        self.save_roc_curve(fpr, tpr)
        
        precision, recall, thresholds = module.precision_recall_curve.cpu().compute()
        self.plot_precision_recall_curve(module, precision, recall)
        self.save_pr_curve(precision, recall)
