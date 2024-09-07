import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib.transforms import Bbox

from workflow.scripts.config import Config


class ResultsPlottingJob:
    def __init__(self, logger):
        self.logger = logger
    
    def blend_with_white(self, color, blend_factor=0.5):
        rgba = np.array(mcolors.to_rgba(color))
        white = np.array([1, 1, 1, 1])
        blended = rgba * blend_factor + white * (1 - blend_factor)
        return tuple(blended)  # Convert back to a tuple
    
    def get_fig_box(self, fig):
        fig.canvas.draw()
        fig_box = fig.get_tightbbox()
        
        margin_px = 1
        margin_in_inches = margin_px / fig.dpi
        
        fig_box_expanded = Bbox.from_bounds(
            fig_box.x0 - margin_in_inches,
            fig_box.y0 - margin_in_inches,
            fig_box.width + 2 * margin_in_inches,
            fig_box.height + 2 * margin_in_inches
        )
        
        return fig_box_expanded
    
    def run(self):
        # results = pd.read_csv(f'{Config.data_dir}/results.csv')
        # results = pd.read_csv(f'/home/user/projects/strats/outputs/results.csv')
        results = pd.read_csv(f'/home/user/projects/thesis_code/results/results.csv')
        results = results[
            (results['dataset'] == 'mimic_iii')
            & (results['train_frac'].between(0.1, 1))
            & (results['model'].isin(['strats', 'gru', 'grud', 'sand', 'tcn']))
            ]
        results['training_time'] = results['training_time'] / 60
        # how much time it took to train strats with 50% of data
        times = results.groupby(['model', 'train_frac'])['training_time']
        print(times.mean()['strats'])
        print(times.std()['strats'])
        
        api = wandb.Api()
        # self.strats_runtimes(api)
        # self.imbalanced_classes_experiments(api)
        fig, axes = plt.subplots(
            1, 3, figsize=(15, 5),
            gridspec_kw={'bottom': 0.2, 'wspace': 0.25, 'top': 0.95, 'left': 0.07, 'right': 0.99})
        auroc_ax, auprc_ax, minrp_ax = axes
        auroc_ax.set_ylabel('AUROC')
        sns.lineplot(x="train_frac",
                     y="test_auroc",
                     data=results,
                     ax=auroc_ax,
                     hue='model',
                     style='model',
                     markers=True)
        auprc_ax.set_ylabel('AUPRC')
        sns.lineplot(x="train_frac",
                     y="test_auprc",
                     data=results,
                     ax=auprc_ax,
                     hue='model',
                     style='model',
                     markers=True)
        minrp_ax.set_ylabel('MinRP')
        sns.lineplot(x="train_frac",
                     y="test_minrp",
                     data=results,
                     ax=minrp_ax,
                     hue='model',
                     style='model',
                     markers=True)
        
        for ax in axes:
            ax.set_xlabel('% labeled data')
            ax.set_xticks(results['train_frac'].unique())
            ax.set_xticklabels([f'{x:,.0%}' for x in ax.get_xticks()])
            # remove legend
            ax.get_legend().remove()
        # add single legend
        handles, labels = auroc_ax.get_legend_handles_labels()
        labels = [label.capitalize() for label in labels]
        fig.legend(handles, labels, loc='lower center', ncol=len(labels))
        # fig.tight_layout(pad=0)
        fig.savefig(f'{Config.data_dir}/plots/baseline_results.pdf',
                    bbox_inches=self.get_fig_box(fig))
        fig.show()
    
    def strats_runtimes(self, wandb_api):
        runs = wandb_api.runs(
            "Strats",
            filters={
                "config.name": "original_strats",
                "config.data_fraction": {"$in": [0.1, 0.5]},
                "created_at": {"$gt": "2024-08-29T00"}
            })
        runtime, name_list, data_fractions = [], [], []
        for run in runs:
            runtime.append(run.summary._json_dict['_runtime'] / 60)
            data_fractions.append(run.config['data_fraction'])
            name_list.append(run.name)
        
        runs_df = pd.DataFrame({
            "runtime": runtime,
            "name": name_list,
            "data_fraction": data_fractions
        })
        return runs_df.groupby(['name', 'data_fraction']).agg(['mean', 'std'])
    
    def imbalanced_classes_experiments(self, wandb_api):
        
        experiments = {
            'original_strats': 'Original (wBCE)',
            'finetune-class-balancing': 'Rebalanced',
            'finetune-larger-batch': 'Batch size 64',
            'finetune-no-weighted-loss': 'Unweighted BCE',
            'finetune-small-batch': 'Batch size 4'
        }
        
        runs = wandb_api.runs(
            "Strats",
            filters={
                "config.name": {"$in": list(experiments.keys())},
                "config.data_fraction": {"$in": [0.1, 0.5]},
                "summary_metrics.test_epoch_mean_prediction": {"$exists": True},
            })
        data = []
        for run in runs:
            json_dict = run.summary._json_dict
            data.append({
                "Experiment": experiments[run.name],
                "Data fraction": f"{run.config['data_fraction']:.0%}".replace('%', r'\%'),
                "auroc": json_dict['test_auroc'],
                "auprc": json_dict['test_pr_auc'],
                "minrp": json_dict['test_minrp'],
                "pred_diff": json_dict['test_epoch_mean_prediction'] - json_dict[
                    'test_pos_class_frac'],
            })
        
        runs_df = pd.DataFrame(data)
        grouped = runs_df.groupby(['Data fraction', 'Experiment']).agg(['mean', 'std']).round(3)
        formatted_df = grouped.copy()
        
        def bold(x):
            return f"\\textbf{{{x}}}"
        
        for metric, new_name, best in [
            ('auroc', r'AUROC \textuparrow', 'max'),
            ('auprc', r'AUPRC \textuparrow', 'max'),
            ('minrp', r'MinRP \textuparrow', 'max'),
            ('pred_diff', r'Pred. Diff. \textbar \textdownarrow \textbar', lambda x: x.loc[x.abs().idxmin()]),
        ]:
            formatted_df[new_name] = formatted_df \
                .apply(lambda row: fr"{row[(metric, 'mean')]}\(\pm\){row[(metric, 'std')]}",
                       axis=1)
            best_vals = grouped[(metric, 'mean')].groupby('Data fraction').transform(best)
            
            best_loc = grouped[(metric, 'mean')] == best_vals
            
            formatted_df.loc[best_loc, new_name] = formatted_df.loc[best_loc, new_name].apply(bold)
            formatted_df.drop(columns=[metric], inplace=True)
        
        formatted_df.columns = formatted_df.columns.droplevel(1)
        # Export the DataFrame to LaTeX
        latex_table = formatted_df.to_latex(
            multirow=True,
            multicolumn=True,
            escape=False,
            column_format='c{1cm}ccccc',
        )
        
        # Print or save the LaTeX table
        print(latex_table)


if __name__ == '__main__':
    logger = None
    job = ResultsPlottingJob(logger)
    job.run()
