import re

import dpath
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from math import ceil

from src.util.data_module import *  # FIXME: remove
from workflow.scripts.config import Config
from workflow.scripts.plotting_functions import get_fig_box


#  TODO: either rename or move table generation to a separate script (latex job?)
class ResultsPlottingJob:
    RUN_KINDS = {
        'VariableStandardScaler': 'z',
        'VariableECDFScaler': 'ecdf',
    }
    
    BASELINE_MODELS = {
        'strats': 'STraTS (orig.)',
        'gru': 'GRU',
        'grud': 'GRU-D',
        'sand': 'SAND',
        'tcn': 'TCN'
    }
    
    NOISE_EXPERIMENTS = {
        "ours": "Z-score",
        "ecdf": "ECDF",
        "ours-noise-4": "Z-score 25% Unif",
        "ours-noise-5": "Z-score 50% Unif",
        "ours-noise-8": r"Z-score $2\sigma$ Gauss",
        "ours-noise-9": "Z-score 75% Unif",
        "ours-noise-10": r"Z-score $3\sigma$ Gauss",
        "ours-noise-11": r"Z-score $1\sigma$ Gauss",
        "ours-noise-12": r"Z-score 100% Unif",
        "ecdf-noise-4": "ECDF 25% Unif",
        "ecdf-noise-5": r"ECDF 50% Unif",
        "ecdf-noise-8": r"ECDF $2\sigma$ Gauss",
        "ecdf-noise-9": "ECDF 75% Unif",
        "ecdf-noise-10": r"ECDF $3\sigma$ Gauss",
        "ecdf-noise-11": r"ECDF $1\sigma$ Gauss",
        "ecdf-noise-12": r"ECDF 100% Unif",
    }
    
    IMBALANCED_CLASSES_EXPERIMENTS = {
        'ours-unweighted': r'$\beta_{16}$',
        'ours-rebalanced': r'$\beta_{16} + os$',
        'ours': r'$\beta_{16} + w + c$',
        "ours-small-batch": r'$\beta_{4} + w + c$',
        "ours-small-batch-no-clipping": r'$\beta_{4} + w$',
        # 'ecdf-clip-small-batch': 'bs4+w+c+ecdf',
    }
    
    def __init__(self, logger):
        self.logger = logger
    
    def run(self):
        api = wandb.Api()
        # self.baselines(api)
        self.noise_experiments(api)
        # self.ablation_experiments(api)
        # self.normalizations_experiments(api)
        # self.imbalanced_classes_experiments(api)
        # self.runtimes(api)
    
    def runtimes(self, api):
        results = self._get_strats_repo_results()
        
        results = results[results['name'] == 'STraTS (orig.)']
        times = results.groupby(['name', 'data_fraction'])['training_time']
        print(times.agg(['mean', 'sem']))
        
        runs_df = self._get_runs_summary(api, {'ours': 'Ours'}, stage='finetune')
        result = runs_df.groupby(['name', 'data_fraction'])['training_time'].agg(['mean', 'sem'])
        print(result)
    
    def baselines(self, api):
        their_results = self._get_strats_repo_results()
        
        our_models = {
            # "original-strats": "original",  # uncomment for comparison and validation
            "ours": 'Ours',
        }
        our_results = self._get_runs_summary(
            api,
            our_models,
            stage='finetune',
            fraction_range=(0.1, 1.0))
        # results = our_results
        results = pd.concat([their_results, our_results])  # FIXME: uncomment
        results = results.dropna(axis=1, how='all')
        
        print(results.groupby(['name', 'data_fraction'])[
                  ['test_auroc', 'test_pr_auc', 'test_minrp']].mean())
        
        axes, fig = self.plot_finetune_metrics(
            results.replace('Ours', 'STraTS (ours)'),
            metrics=('test_auroc', 'test_pr_auc', 'test_minrp'),
            style='name', hue='name', markers=True
        )
        fig.tight_layout()
        fig.show()
        fig.savefig(f'{Config.data_dir}/plots/baseline_results.pdf',  # FIXME: use output property
                    bbox_inches=get_fig_box(fig))
        
        # print table
        baseline = 'STraTS (orig.)'
        # table_experiments = (baseline, 'Ours')
        table_experiments = results['name'].unique()
        table_results = results[
            results['data_fraction'].isin([0.1, 0.5, 1.0])
            #  & results['name'].isin(table_experiments)
        ]
        table_results.drop(columns='weighted_test_loss', inplace=True)
        table = self._format_table(
            table_results,
            group_by=['data_fraction', 'name'],
            baseline=baseline)
        latex = self._to_latex(
            table,
            table_experiments,
            column_format='p{1.5cm}lc...',
            baseline=baseline,
        )
        print(latex)
    
    def ablation_experiments(self, api):
        exps = {
            'ecdf': 'Original',
            'no-cve_value': 'No value embedding',
            'no-cve_time': 'No time embedding',
            'no-variable-emb': 'No variable embedding',
            'ours-noise-12': '100% Unif noise',
            'ours-noise-13': 'Shuffle feature types',
            'no-demographics': 'No demographics',
            'no-timeseries': 'No timeseries',
        }
        results = self._get_runs_summary(api, exps, stage='finetune', fraction_range=(1.0, 1.0))
        table = self._format_table(
            results,
            group_by=['name'],
            bold_best=False,
            baseline='Original'
        )
        latex = self._to_latex(
            table,
            experiment_names=tuple(exps.values()),
            column_format='lc...',
            baseline='Original'
        )
        print(latex)
    
    def noise_experiments(self, api):
        results = self._get_runs_summary(api, self.NOISE_EXPERIMENTS, stage='finetune')
        results.sort_values(['kind', 'noise_type', 'data'], inplace=True)
        
        # results['marker'] = results['kind'].map({'z': 'o', 'ecdf': 'X'})
        markers = results.groupby('name')['kind'].first().map({'z': 'o', 'ecdf': 'X'}).to_dict()
        
        full_noise_exps = {
            self.NOISE_EXPERIMENTS[k] for k in
            ['ours-noise-12', 'ecdf-noise-12']}
        no_noise_exps = {
            self.NOISE_EXPERIMENTS[k] for k in
            ['ours', 'ecdf']}
        are_baselines = (results['name'].isin(no_noise_exps)
                         | results['name'].isin(full_noise_exps))
        
        all_results = results.copy()
        baseline_results = results[are_baselines].copy()
        results = results[~are_baselines].copy()
        
        baseline_palette = {
            exp: (.75, .75, .75) for exp in no_noise_exps | full_noise_exps
        }
        metrics = ('test_auroc', 'test_pr_auc', 'test_minrp')
        
        def plot_noise_results(noise_results, baselines=None):
            # color by data
            datas = noise_results['data'].unique()
            data_colors = dict(zip(datas, sns.color_palette("colorblind", len(datas))))
            palette = (noise_results.groupby('name')['data'].first().map(data_colors).to_dict()
                       | baseline_palette)
            
            axes, fig = self.plot_finetune_metrics(
                pd.concat([baselines, noise_results]),
                metrics=metrics,
                style='name',
                markers=markers,
                palette=palette,
                hue='name',
            )
            fig.set_size_inches(fig.get_size_inches() * [1, 1.33])
            return fig
        
        for noise_type in ['uniform', 'gaussian']:
            noise_results = results[results['noise_type'] == noise_type]
            
            fig = plot_noise_results(noise_results, baseline_results)
            
            fig.show()
            fig.savefig(f'{Config.data_dir}/plots/{noise_type}_noise_results.pdf',
                        bbox_inches=get_fig_box(fig))
        
        table_results = (
            all_results[all_results['data_fraction'] == 1.0]
            .drop(columns=['data_fraction', 'weighted_test_loss'], errors='ignore'))
        
        table = self._format_table(
            table_results,
            group_by=['name'],
            bold_best=False,
            baseline='Z-score'
        )
        latex = self._to_latex(
            table,
            experiment_names=tuple(self.NOISE_EXPERIMENTS.values()),
            column_format='lc...',
            baseline='Z-score'
        )
        print(latex)
        
        # stretch between baselines
        min_max = (
            baseline_results
            .groupby(['data', 'data_fraction'])
            .mean(numeric_only=True)
            .reset_index()
            .groupby(['data_fraction']).agg({m: ('min', 'max') for m in metrics}))
        # Reformat min_max DataFrame to avoid a multi-index
        min_max.columns = [f"{metric}_{ext}" for metric, ext in min_max.columns]
        
        rescaled_results = (
            results[results['noise_type'] == 'uniform'].copy()
            .set_index(['data_fraction']))
        
        for metric in metrics:
            min_col = f"{metric}_min"
            max_col = f"{metric}_max"
            
            # Align with the min_max data by joining on data_fraction
            rescaled_results = rescaled_results.join(
                min_max[[min_col, max_col]], on='data_fraction')
            
            # Perform min-max scaling
            rescaled_results[metric] = (
                (rescaled_results[metric] - rescaled_results[min_col]) /
                (rescaled_results[max_col] - rescaled_results[min_col])
            )
            
            # Drop the min and max columns for cleanliness after scaling
            rescaled_results = rescaled_results.drop(columns=[min_col, max_col])
        rescaled_results.reset_index(inplace=True)
        fig = plot_noise_results(rescaled_results)
        fig.show()
        
        table_results = (
            rescaled_results[rescaled_results['data_fraction'] == 1.0]
            .drop(columns=['data_fraction', 'weighted_test_loss'], errors='ignore'))
        table_results.sort_values(['data', 'name'], inplace=True)
        
        # map data names
        table_results['data'] = (
            table_results['data']
            .map(lambda x: f"{float(x.replace('noisy_uniform_p', '')) * 100}%"))
        table_results['kind'] = table_results['kind'].map({'z': 'Z-score', 'ecdf': 'ECDF'})
        
        table = self._format_table(
            table_results,
            group_by=['data', 'kind'],
            bold_best=True,
            baseline='Z-score'
        )
        table.index.rename({'data': 'Noise level', 'kind': 'Normalization'}, inplace=True)
        
        latex = self._to_latex(
            table,
            experiment_names=tuple(self.NOISE_EXPERIMENTS.values()),
            column_format='lc...',
            baseline='Z-score'
        )
        print(latex)
    
    def imbalanced_classes_experiments(self, api):
        results = self._get_runs_summary(
            api, self.IMBALANCED_CLASSES_EXPERIMENTS, stage='finetune'
        )
        results['pred_diff'] = (results['test_mean_prediction'] - results[
            'test_pos_class_frac']) / results['test_pos_class_frac']
        results = results[results['data_fraction'].isin([0.1, 0.5, 1.0])]
        results.sort_values(['batch_size'], inplace=True)
        formatted_df = self._format_table(results, group_by=['data_fraction', 'name'])
        latex_table = self._to_latex(
            formatted_df,
            tuple(self.IMBALANCED_CLASSES_EXPERIMENTS.values()),
            column_format='p{1.5cm}lc...')
        print(latex_table)
    
    def normalizations_experiments(self, wandb_api):
        baseline = 'Z-score'
        pretrain_experiments = {
            'ours': baseline,
            # 'ecdf': 'ECDF',
            # 'ecdf-no-sigmoid': 'ECDF no sigmoid',
            'ecdf-clip': 'ECDF',
        }
        
        pretrain_runs_df = self._get_runs_summary(wandb_api, pretrain_experiments, stage='pretrain')
        pretrain_runs_df.sort_values(['kind'], inplace=True)
        formatted_df = self._format_table(pretrain_runs_df, group_by=['name'], baseline=baseline)
        latex_table = self._to_latex(
            formatted_df,
            tuple(pretrain_experiments.values()),
            baseline=baseline,
            column_format='lc...')
        print("Pretraining normalization experiment results", latex_table, sep='\n')
        
        finetune_experiments = {
            'ours': baseline,
            'ecdf': 'ECDF',
        }
        finetune_runs_df = self._get_runs_summary(
            wandb_api,
            finetune_experiments,
            stage='finetune'
        )
        axes, fig = self.plot_finetune_metrics(
            finetune_runs_df,
            metrics=('test_auroc', 'test_pr_auc', 'test_minrp'),
            style='name',
            hue='name',
            markers=True,
        )
        fig.show()
        
        finetune_runs_df = finetune_runs_df[finetune_runs_df['data_fraction'].isin([0.1, 0.5, 1.0])]
        finetune_runs_df.drop(columns='weighted_test_loss', inplace=True)
        formatted_df = self._format_table(
            finetune_runs_df,
            group_by=['data_fraction', 'name'],
            baseline=baseline,
        )
        latex_table = self._to_latex(
            formatted_df,
            tuple(finetune_experiments.values()),
            column_format='p{1.5cm}lc...',
            baseline=baseline,
        )
        
        print("Finetuning normalization experiment results", latex_table, sep='\n')
    
    def plot_finetune_metrics(self, results: pd.DataFrame, metrics=None, **kwargs):
        metric_names = {
            'test_auroc': 'ROC-AUC',
            'test_pr_auc': 'AUC-PR',
            'test_minrp': 'min(Re,Pr)',
            # 'test_epoch_loss': 'Test Loss',
        }
        if metrics is None:
            metrics = metric_names.keys()
        
        fig, axes = plt.subplots(
            1, len(metrics), figsize=(len(metrics) * 4.5, 5),
            gridspec_kw={'bottom': 0.2, 'wspace': 0.25, 'top': 0.95, 'left': 0.07, 'right': 0.99}
        )
        for col, ax in zip(metrics, axes):
            if col not in metrics:
                continue
            
            metric_name = metric_names[col]
            self.plot_finetune_metric(results, col=col, metric_name=metric_name, ax=ax, **kwargs)
        
        # add single legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        ncol = len(handles) if len(handles) < 7 else ceil(len(handles) / 2)
        fig.legend(handles=handles, labels=labels, loc='lower center', ncol=ncol)
        return axes, fig
    
    def plot_finetune_metric(
        self, df: pd.DataFrame, col: str, metric_name: str, ax: plt.Axes,
        **kwargs,
    ):
        kwargs = {
                     'x': 'data_fraction',
                     'y': col,
                     # 'palette': 'colorblind',
                     'estimator': 'mean',
                     'data': df,
                     'ax': ax,
                     'errorbar': 'se',
                     'legend': True,  # plot legend so we can reuse the handles
                 } | kwargs
        
        sns.lineplot(**kwargs)
        
        ax.set_ylabel(metric_name)
        ax.set_xlabel('Data Fraction')
        ax.set_xticks(df['data_fraction'].unique())
        ax.set_xticklabels([f'{x:,.0%}' for x in ax.get_xticks()])
        # remove legend
        ax.get_legend().remove()
    
    def _format_table(
        self,
        df: pd.DataFrame,
        group_by: list[str],
        baseline=None,
        bold_best=True
    ) -> pd.DataFrame:
        """
        Use bold font for the best value in each column
        Group the results by data fraction and experiment name
        Show the mean and standard error of the metrics
        :param df:
        :return:
        """
        
        metrics = {
            'test_auroc': (r'ROC-AUC \textuparrow', 'max', r'{mean:.3f}\(\pm\){sem:.3f}'),
            'test_pr_auc': (r'AUC-PR \textuparrow', 'max', r'{mean:.3f}\(\pm\){sem:.3f}'),
            'test_minrp': (r'min(Re,Pr) \textuparrow', 'max', r'{mean:.3f}\(\pm\){sem:.3f}'),
            'pred_diff': (r'MPD \textbar \textdownarrow \textbar',
                          lambda x: x.loc[x.abs().idxmin()], '{mean:+.1%}'),
            # 'weighted_test_loss': (  # TODO: consider adding for imbalanced experiments
            #     r'wBCE \textdownarrow', 'min', r'{mean:.2f}\(\pm\){sem:.2f}'),
            'test_standard_mse_loss': (
                r'Z MSE $\times 10^3$ \textdownarrow', 'min', r'{mean:.1f}\(\pm\){sem:.1f}'),
            'test_standard_mae_loss': (
                r'Z MAE $\times 10^3$ \textdownarrow', 'min', r'{mean:.1f}\(\pm\){sem:.1f}'),
            'test_ecdf_mse_loss': (
                r'E MSE $\times 10^3$ \textdownarrow', 'min', r'{mean:.1f}\(\pm\){sem:.1f}'),
            'test_ecdf_mae_loss': (
                r'E MAE $\times 10^3$ \textdownarrow', 'min', r'{mean:.1f}\(\pm\){sem:.1f}'),
            # 'epoch': (r'Epoch \textdownarrow', 'min'),
        }
        
        if (df['stage'].unique() == ['pretrain']).all():
            # multiply pretraining loss columns by 1000
            loss_re = re.compile(r'_loss')
            loss_cols = [col for col in df.columns if loss_re.search(col)]
            df[loss_cols] *= 1000
        
        df = (df
              .drop(columns=list(set(df.columns) - (metrics.keys() | set(group_by))))
              .dropna(axis=1, how='all'))
        
        grouped = df.groupby(group_by, sort=False) \
            .agg(['mean', 'sem']).round(3)
        
        if baseline is not None:
            for metric in df.columns:
                if metric in group_by:
                    continue
                baseline_mean = grouped.loc[
                    grouped.index.get_level_values(group_by[-1]) == baseline, (
                        metric, 'mean')]
                if baseline_mean.index.nlevels > 1:
                    baseline_mean = baseline_mean.droplevel(-1)
                else:
                    baseline_mean = baseline_mean.iloc[0]
                grouped[(metric, 'rel_diff')] = (grouped[(
                    metric, 'mean')] - baseline_mean) / baseline_mean
        
        formatted_df = grouped.copy()
        
        def bold(x):
            return f"\\textbf{{{x}}}"
        
        for metric, (new_name, best, str_format) in metrics.items():
            if metric not in formatted_df.columns:
                continue
            
            def format_cell(row):
                ...
                if pd.isna(row[metric]).all():
                    return ''
                if baseline is not None and not (
                    row.name == baseline or isinstance(row.name, tuple) and row.name[-1] == baseline
                ):
                    return (str_format + r" \({rel_diff:+.0%}\)").format(**row[metric])
                else:
                    return str_format.format(**row[metric])
            
            formatted_df[new_name] = formatted_df \
                .apply(format_cell, axis=1)
            
            if bold_best:
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
    
    def _to_latex(
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
            # assert df.index.names == ['data_fraction', 'name']
            df.index.rename({'data_fraction': 'Data fraction', 'name': ' ', 'kind': ' ',
                             'data': 'Data'}, inplace=True)
            
            # name_index = pd.CategoricalIndex(
            #     data=df.index.levels[1], categories=experiment_names)
            # frac_index = pd.CategoricalIndex(
            #     data=df.index.levels[0], categories=sorted(df.index.levels[0]))
            # df.index = df.index.set_levels((
            #     frac_index,
            #     name_index
            # ))
            # df.sort_index(inplace=True)
            # df.index = df.index.set_levels((
            #     [f"{x:.0%}" for x in df.index.levels[0]],
            #     df.index.levels[1]
            # ))
            
            # underscore baseline names
            if baseline is not None:
                new_index = [(i[0], (underscored(i[1]) if i[1] == baseline else i[1])) for i in
                             df.index]
                df.index = pd.MultiIndex.from_tuples(new_index)
        
        else:
            assert df.index.name == 'name'
            df.index = pd.CategoricalIndex(
                name=' ', data=df.index,
                categories=experiment_names)
            # df.sort_index(inplace=True)
            
            if baseline is not None:
                df.index = df.index.map(lambda x: underscored(x) if x == baseline else x)
        
        if column_format.endswith('...'):
            # remove the '...' suffix and '{anything}'
            column_format = column_format.rstrip('...')
            stripped = re.sub(r"\{.*?}", "", column_format)
            
            # fill the column format with the last element
            last_col_format = stripped[-1]
            column_format += last_col_format * (df.shape[1] + df.index.nlevels - len(stripped))
        
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
    
    def _get_strats_repo_results(self):
        results = pd.read_csv(f'results/results.csv')
        results['training_time'] = results['training_time'] / 60
        results['name'].replace(self.BASELINE_MODELS, inplace=True)
        return results
    
    def _get_runs_summary(
        self,
        api: wandb.Api,
        models: dict[str, str],
        stage: Literal['pretrain', 'finetune'],
        fraction_range: tuple[float, float] | None = None
    ) -> pd.DataFrame:
        filters = {
            "display_name": {"$in": list(models.keys())},
            'tags': {"$nin": ['archive', 'with-bias']},  # fixme: uncomment
            'state': 'finished',
        }
        if fraction_range is not None:
            filters['config.data_fraction'] = {"$gte": fraction_range[0], "$lte": fraction_range[1]}
        
        if stage is not None:
            filters['config.stage'] = stage
        
        runs = api.runs(filters=filters)
        
        our_data = []
        noise_re = re.compile(r'noisy_([a-z]+)')
        
        for run in runs:
            run: wandb.apis.public.Run
            summary_dict = run.summary._json_dict
            
            kind = self.RUN_KINDS[
                dpath.get(run.config, 'data_config/train/dataset/scaler_class')]
            data = dpath.get(run.config, 'data_config/train/dataset/path').split('/')[-2]
            
            our_data.append({
                **run.config,
                **summary_dict,
                "display_name": run.display_name,
                "kind": kind,
                "data": data,
                "batch_size": dpath.get(run.config, 'data_config/train/loader/batch_size'),
                'noise_type': match.group(1) if (match := noise_re.search(data)) else 'none',
                "name": models[run.name],
                "training_time": summary_dict['_runtime'] / 60,
                'epoch': summary_dict['epoch'] -
                         dpath.get(run.config, 'early_stop_callback/patience', default=0),
            })
        kind_dtype = pd.CategoricalDtype(categories=['z', 'ecdf'], ordered=True)
        noise_dtype = pd.CategoricalDtype(categories=['none', 'gaussian', 'uniform'], ordered=True)
        df = pd.DataFrame(our_data).astype({
            'kind': kind_dtype,
            'noise_type': noise_dtype,
        })
        
        return df


if __name__ == '__main__':
    logger = None
    job = ResultsPlottingJob(logger)
    job.run()
