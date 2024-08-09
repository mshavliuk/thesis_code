import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from workflow.scripts.config import Config


class ResultsPlottingJob:
    def __init__(self, logger):
        self.logger = logger
    
    def run(self):
        # results = pd.read_csv(f'{Config.data_dir}/results.csv')
        results = pd.read_csv(f'/home/user/projects/strats/outputs/results.csv')
        results = results[
            (results['dataset'] == 'mimic_iii')
            & (results['train_frac'].between(0.1, 1))
            ]
        
        fig, axes = plt.subplots(
            1, 3, figsize=(15, 5),
            gridspec_kw={'bottom': 0.2, 'wspace': 0.25, 'top': 0.95, 'left': 0.07, 'right': 0.99})
        auroc_ax, auprc_ax, minrp_ax = axes
        auroc_ax.set_ylabel('AUROC')
        sns.lineplot(x="train_frac", y="test_auroc", data=results, ax=auroc_ax, hue='model')
        auprc_ax.set_ylabel('AUPRC')
        sns.lineplot(x="train_frac", y="test_auprc", data=results, ax=auprc_ax, hue='model')
        minrp_ax.set_ylabel('MinRP')
        sns.lineplot(x="train_frac", y="test_minrp", data=results, ax=minrp_ax, hue='model')
        for ax in axes:
            ax.set_xlabel('% labeled data')
            ax.set_xticks(results['train_frac'].unique())
            ax.set_xticklabels([f'{x:,.0%}' for x in ax.get_xticks()])
            # remove legend
            ax.get_legend().remove()
        # add single legend
        handles, labels = auroc_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3)
        fig.tight_layout()
        fig.savefig(f'{Config.data_dir}/plots/results.eps')
        plt.show()


if __name__ == '__main__':
    logger = None
    job = ResultsPlottingJob(logger)
    job.run()
