import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
import wandb
from matplotlib import pyplot as plt

from src.util.common import (
    create_data_module,
    create_model_module,
    create_trainer,
    setup,
)
from src.util.config import (
    parse_config,
)
from src.util.wandb import (
    find_checkpoint,
    get_run_checkpoint,
)
from workflow.scripts.util import get_fig_box


def plot_risk_score_distributions(config, run: wandb.apis.public.Run, ax: list[plt.Axes]):
    data_checkpoint, data_artifact = find_checkpoint(config, checkpoint_type='data_module')
    logger = logging.getLogger(__name__)
    ax_common_norm, ax_self_norm = ax
    
    data = create_data_module(config, logger, data_checkpoint)
    
    checkpoint = get_run_checkpoint(run, type='model')
    model = create_model_module(config, logger, data, checkpoint)
    
    trainer = create_trainer(config)
    preds = trainer.predict(model, data.test_dataloader())
    preds = torch.sigmoid(torch.vstack(preds).squeeze())
    
    truths = torch.vstack([b['label'] for b in data.test_dataloader()]).squeeze()
    df = pd.DataFrame({
        'risk_score': preds.float().numpy(),
        'label': truths.bool().cpu().numpy(),
    })
    # plot distributions of risk scores by class
    kde_kws = {
        'data': df,
        'x': 'risk_score',
        'hue': 'label',
        'fill': True,
        'clip': (0, 1),
        'legend': True,
    }
    sns.kdeplot(**kde_kws, common_norm=True, ax=ax_common_norm)
    
    sns.kdeplot(**kde_kws, common_norm=False, ax=ax_self_norm)
    ax_self_norm.set_xlabel('Predicted risk')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target', '-t', type=str, required=True, help='Run ID of the model to renew', nargs='+')
    args = parser.parse_args()
    api = wandb.Api()
    
    setup()
    
    risk_dist_path = Path(os.environ['DATA_DIR'] + '/plots').resolve()
    risk_dist_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(
        2,
        len(args.target),
        sharex=True,
        figsize=(4 * len(args.target), 8),
        gridspec_kw={'bottom': 0.12, 'wspace': 0.1, 'top': 0.95, 'left': 0.07, 'right': 0.99, 'hspace': 0.1},
    )
    
    names_map = {
        'ours-small-batch': r'$\beta{4}+w+c$',
        'ours-small-batch-no-clipping': r'$\beta{4}+w$',
        'ours': r'$\beta{16}+w+c$',
        'ours-rebalanced': r'$\beta{16} + os$',
    }
    
    for run_id, col_ax in zip(args.target, axes.T):
        run = api.run(run_id)
        config = parse_config(run.config)
        plot_risk_score_distributions(config, run, col_ax)
        col_ax[0].set_title(names_map[run.config['name']])
    
    legend: plt.Legend = axes.ravel()[0].get_legend()
    handles = legend.legend_handles
    fig.legend(handles=handles, labels=['Survived', 'Died'], loc='lower center', ncol=len(handles))
    
    for ax in axes.ravel():
        ax.set_ylabel('')
        ax.get_legend().remove()
    
    axes[0, 0].set_ylabel('Density (Global)')
    axes[1, 0].set_ylabel('Density (Per Class)')
    
    fig.tight_layout()
    fig.show()
    fig.savefig(risk_dist_path / f"risk_score_distributions.pdf", bbox_inches=get_fig_box(fig))


if __name__ == '__main__':
    main()
