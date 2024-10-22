import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

from workflow.scripts.config import Config
from workflow.scripts.util import get_fig_box

PROJECT_NAME = 'Strats'  # todo: get from env


#  TODO: either rename or move table generation to a separate script (latex job?)
class ResultsPlottingJob:
    NORMALIZATION_EXPERIMENTS = OrderedDict([
        ('pretrain-ours', 'Z-score'),
        ('pretrain-xl', 'Z-score XL'),
        ('pretrain-xxl', 'Z-score XXL'),
        ('pretrain-ecdf', 'ECDF'),
        ('pretrain-xl-ecdf', 'ECDF XL'),
        ('pretrain-xxl-ecdf', 'ECDF XXL'),
    ])
    
    def __init__(self, logger):
        self.logger = logger
    
    def run(self):
        api = wandb.Api()
        self.baselines(api)
        exit()
        self.imbalanced_classes_experiments(api)
        self.normalizations_experiments(api)
        self.runtimes(api)
        self.multiphase_training_experiments_plot(api)
        self.multiphase_training_experiments_table(api)
    
    def get_runs_summary(self, api, models: dict[str, str], fractions=None) -> pd.DataFrame:
        runs = api.runs(
            PROJECT_NAME,
            filters={
                '$or': [
                    {"config.name": {"$in": list(models.keys())}},
                    {"display_name": {"$in": list(models.keys())}},
                ],
                '$and': [
                            {'tags': {"$nin": ['archive', 'with-bias']}},
                            {"state": {"$ne": "running"}},
                        ] + ([
                                 {"config.data_fraction": {"$in": fractions}}
                             ] if fractions is not None else []),
            })
        our_data = []
        for run in runs:
            json_dict = run.summary._json_dict
            our_data.append({
                **run.config,
                **json_dict,
                "name": models[run.name],
                "training_time": json_dict['_runtime'] / 60,
                'epoch': json_dict['epoch'] - run.config.get('early_stop_callback', {}).get('patience', 0),
            })
        df = pd.DataFrame(our_data)
        
        return df
    
    def baselines(self, api):
        # results = pd.read_csv(f'{Config.data_dir}/results.csv')
        # results = pd.read_csv(f'/home/user/projects/strats/outputs/results.csv')
        results = pd.read_csv(
            f'/home/user/projects/thesis_code/results/results.csv',
            usecols=['model', 'train_frac', 'test_auroc', 'test_auprc', 'test_minrp',
                     'training_time']
        ).rename(columns={'model': 'name', 'train_frac': 'data_fraction',
                          'test_auprc': 'test_pr_auc'})
        baseline_models = {
            # 'strats': 'STraTS (orig.)',
            # 'gru': 'GRU', # FIXME: uncomment
            # 'grud': 'GRU-D',
            # 'sand': 'SAND',
            # 'tcn': 'TCN'
        }
        results = results[
            (results['data_fraction'].between(0.1, 1))
            & (results['name'].isin(list(baseline_models.keys())))
            ]
        results['training_time'] = results['training_time'] / 60
        # rename models
        results['name'].replace(baseline_models, inplace=True)
        # how much time it took to train strats with 50% of data
        times = results.groupby(['name', 'data_fraction'])['training_time']
        print(times.agg(['mean', 'std']))
        print(results.groupby(['name', 'data_fraction'])[
                  ['test_auroc', 'test_pr_auc', 'test_minrp']].agg(['mean', 'std']))
        
        our_models = {
            # "finetune-original-strats": "original",
            # "finetune-with-outliers": "ad+clip outliers",
            "finetune-ours": "ad+clip",
            "finetune-ours-noise-1": "ours gauss noise",
            "finetune-ours-noise-2": "ours unif noise",
            # "finetune-ecdf-sigmoid-ad": "ecdf sigmoid ad",
            # "finetune-ecdf-sigmoid-ad-outliers": "ecdf sigmoid outliers + ad",
        }
        our_results = self.get_runs_summary(
            api, our_models,
            fractions=[float(f"0.{i+1}") for i in range(9)] + [1.0]
        )
        results = pd.concat([results, our_results])
        results = results.dropna(axis=1, how='all')
        
        print(results.groupby(['name', 'data_fraction'])[
            ['test_auroc', 'test_pr_auc', 'test_minrp']].mean())
        
        fig, axes = plt.subplots(
            1, 3, figsize=(15, 5),
            gridspec_kw={'bottom': 0.2, 'wspace': 0.25, 'top': 0.95, 'left': 0.07, 'right': 0.99})
        auroc_ax, auprc_ax, minrp_ax = axes
        
        for col, metric, ax in (
            ('test_auroc', 'ROC-AUC', auroc_ax),
            ('test_pr_auc', 'AUC-PR', auprc_ax),
            ('test_minrp', 'min(Re,Pr)', minrp_ax)
        ):
            self.plot_finetune_metric(results, col, metric, ax)
        
        # add single legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(labels))
        fig.show()
        fig.savefig(f'{Config.data_dir}/plots/baseline_results.pdf',  # FIXME: use output property
                    bbox_inches=get_fig_box(fig))
        
        # print table
        baseline = 'ad+clip'
        table_results = results[results['data_fraction'].isin([0.1, 0.5, 1.0])]
        table = self.format_table(table_results,
                                  group_by=['data_fraction', 'name'],
                                  baseline=baseline)
        experiment_names = tuple(baseline_models.values()) + tuple(our_models.values())
        latex = self.to_latex(table, experiment_names, column_format='p{1.5cm}lcccc', baseline=baseline)
        print(latex)
    
    def plot_finetune_metric(self, df: pd.DataFrame, col: str, metric_name: str, ax: plt.Axes):
        sns.lineplot(
            x="data_fraction",
            estimator='mean',
            n_boot=10_000,
            y=col,
            data=df,
            ax=ax,
            # errorbar='se',
            errorbar=('ci', 90),
            hue='name',
            style='name',
            markers=True,
            legend=True,  # plot legend so we can reuse the handles
        )
        ax.set_ylabel(metric_name)
        ax.set_xlabel('% labeled data')
        ax.set_xticks(df['data_fraction'].unique())
        ax.set_xticklabels([f'{x:,.0%}' for x in ax.get_xticks()])
        # remove legend
        ax.get_legend().remove()
    
    def runtimes(self, wandb_api):
        runs_df = self.get_runs_summary(wandb_api, {'finetune-ours': 'STraTS (ours)'})
        result = runs_df.groupby(['name', 'data_fraction']).agg(['mean', 'std'])
        print(result['training_time'])
        return result
    
    def format_table(self, df, group_by: list[str], baseline=None) -> pd.DataFrame:
        """
        Use bold font for the best value in each column
        Group the results by data fraction and experiment name
        Show the mean and standard deviation of the metrics
        :param df:
        :return:
        """
        # select only numeric columns
        df = df.set_index(group_by).select_dtypes(include='number').reset_index()
        grouped = df.groupby(group_by) \
            .agg(['mean', 'std']).round(3)
        # add cv column and reldiff compared to baseline
        for metric in df.columns:
            if metric in group_by:
                continue
            grouped[(metric, 'cv')] = (grouped[(metric, 'std')] / grouped[(metric, 'mean')]).abs()
            if baseline is not None:
                baseline_mean = grouped.loc[
                    grouped.index.get_level_values(group_by[-1]) == baseline, (metric, 'mean')]
                # drop the last level of the index so that we can use the same index for the new column
                baseline_mean.index = baseline_mean.index.droplevel(-1)
                grouped[(metric, 'rel_diff')] = (grouped[(
                metric, 'mean')] - baseline_mean) / baseline_mean
        
        formatted_df = grouped.copy()
        
        def bold(x):
            return f"\\textbf{{{x}}}"
        
        metrics = {
            'test_auroc': (r'ROC-AUC \textuparrow', 'max'),
            'test_pr_auc': (r'AUC-PR \textuparrow', 'max'),
            'test_minrp': (r'min(Re,Pr) \textuparrow', 'max'),
            'pred_diff': (r'Pred. Diff. \textbar \textdownarrow \textbar',
                          lambda x: x.loc[x.abs().idxmin()]),
            'test_epoch_standardized_mse_loss': (r'Z MSE Loss \textdownarrow', 'min'),
            'test_epoch_standardized_mae_loss': (r'Z MAE Loss \textdownarrow', 'min'),
            'test_epoch_ecdf_mse_loss': (r'ECDF MSE Loss \textdownarrow', 'min'),
            'test_epoch_ecdf_mae_loss': (r'ECDF MAE Loss \textdownarrow', 'min'),
            'epoch': (r'Epoch \textdownarrow', 'min'),
        }
        
        for metric, (new_name, best) in metrics.items():
            if metric not in df.columns:
                continue
            
            if baseline is None:
                if metric == 'pred_diff':
                    str_format = r"{mean:+.1%}"
                else:
                    str_format = r"{mean:.3f}\(\pm\){cv:.0%}"
            else:
                if metric == 'pred_diff':
                    str_format = r"{mean:+.1%} \({rel_diff:+.1%}\)"
                else:
                    str_format = r"{mean:.3f}\(\pm\){cv:.0%} \({rel_diff:+.1%}\)"
            
            
            formatted_df[new_name] = formatted_df \
                .apply(lambda row: ('' if pd.isna(row[metric]).all() else str_format.format(**row[metric])), axis=1)
            if len(group_by) > 1:
                best_vals = grouped[(metric, 'mean')].groupby(group_by[0]).transform(best)
            elif isinstance(best, str):
                best_vals = getattr(grouped[(metric, 'mean')], best)()
            else:
                raise NotImplementedError
            best_loc = grouped[(metric, 'mean')] == best_vals
            
            formatted_df.loc[best_loc, new_name] = formatted_df.loc[
                best_loc, new_name].apply(bold).values
            
            formatted_df.drop(columns=[metric], inplace=True)
        
        formatted_df.columns = formatted_df.columns.droplevel(1)
        formatted_df.drop(columns=set(formatted_df.columns) & (
            set(df.columns) - set(metrics.keys())), inplace=True)
        
        return formatted_df
    
    def to_latex(
        self,
        df: pd.DataFrame,
        experiment_names: tuple[str, ...],
        column_format: str,
        baseline=None
    ) -> str:
        df = df.copy()
        
        def underscored(x):
            return f"\\underline{{{x}}}"
        
        if isinstance(df.index, pd.MultiIndex):
            assert df.index.names == ['data_fraction', 'name']
            name_index = pd.CategoricalIndex(
                data=df.index.levels[1], categories=experiment_names)
            frac_index = pd.CategoricalIndex(
                data=df.index.levels[0], categories=sorted(df.index.levels[0]))
            df.index = df.index.set_levels((
                frac_index,
                name_index
            ))
            df.sort_index(inplace=True)
            df.index = df.index.set_levels((
                [f"{x:.0%}" for x in df.index.levels[0]],
                df.index.levels[1]
            ))
            
            # underscore baseline names
            if baseline is not None:
                new_index = [(i[0], (underscored(i[1]) if i[1] == baseline else i[1])) for i in
                             df.index]
                df.index = pd.MultiIndex.from_tuples(new_index)
            df.index.set_names(['Data fraction', 'Experiment'], inplace=True)
        
        else:
            assert df.index.name == 'name'
            df.index = pd.CategoricalIndex(
                name='Experiment', data=df.index,
                categories=experiment_names)
            df.sort_index(inplace=True)
            
            if baseline is not None:
                df.index = df.index.map(lambda x: underscored(x) if x == baseline else x)
        
        latex_table = df.to_latex(
            multirow=True,
            multicolumn=True,
            escape=False,
            column_format=column_format,
        ).replace('%', r'\%')
        
        headers_re = re.compile(
            r'^\\toprule$\n^(?P<header>.*)\\\\$\n(?P<index>.*)\\\\$\n^\\midrule$',
            re.MULTILINE)
        match = headers_re.search(latex_table)
        header_list = match.group('header').split(' & ')
        index_list = match.group('index').split(' & ')
        
        # Merge the two lists: prioritize non-empty elements from the index list
        merged_list = [h + i for h, i in zip(header_list, index_list)]
        
        # replace
        replacement = fr"\\toprule\n{' & '.join(merged_list).replace('\\', r'\\')} \\\\\n\\midrule"
        latex_table = headers_re.sub(replacement, latex_table)
        return latex_table
    
    def imbalanced_classes_experiments(self, wandb_api):
        experiments = {
            'finetune-no-weighted-loss': 'bs16',
            'finetune-class-balancing': 'bs16 50/50',
            'finetune-ours': 'bs16+w',
            'finetune-ours-clipped': 'bs16+w+c',
            "finetune-small-batch": 'bs4+w',
            "finetune-small-batch-clipped": 'bs4+w+c',
        }
        runs_df = self.get_runs_summary(wandb_api, experiments, fractions=[0.1, 0.5])
        runs_df['pred_diff'] = (runs_df['test_epoch_mean_prediction'] - runs_df[
            'test_pos_class_frac']) / runs_df['test_pos_class_frac']

        formatted_df = self.format_table(runs_df, group_by=['data_fraction', 'name'])
        latex_table = self.to_latex(formatted_df, experiments, column_format='p{3cm}lcccc')
        print(latex_table)
    
    def normalizations_experiments(self, wandb_api):
        experiments = OrderedDict([
            ('original-strats', 'Orig'),
            ('ours', 'Z-score'),
            ('xl', 'Z-score XL'),
            ('xxl', 'Z-score XXL'),
            ('ecdf', 'ECDF'),
            ('xl-ecdf', 'ECDF XL'),
            ('xxl-ecdf', 'ECDF XXL'),
        ])
        pretrain_experiments = {f"pretrain-{key}": value for key, value in experiments.items()}
        pretrain_runs_df = self.get_runs_summary(wandb_api, pretrain_experiments)
        formatted_df = self.format_table(pretrain_runs_df, group_by=['name'])
        latex_table = self.to_latex(formatted_df, experiments, column_format='lcccc')
        print("Pretraining normalization experiment results", latex_table, sep='\n')
        
        finetune_experiments = {f"finetune-{key}": value for key, value in experiments.items()}
        finetune_runs_df = self.get_runs_summary(
            wandb_api, finetune_experiments, fractions=[0.1, 0.5, 1.0])
        formatted_df = self.format_table(finetune_runs_df, group_by=['data_fraction', 'name'])
        latex_table = self.to_latex(formatted_df, experiments, column_format='p{3cm}lccc')
        
        print("Finetuning normalization experiment results", latex_table, sep='\n')


if __name__ == '__main__':
    logger = None
    job = ResultsPlottingJob(logger)
    job.run()
