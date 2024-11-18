import argparse
import os
import re
from functools import reduce

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import (
    colors,
    pyplot as plt,
)
from pyspark.sql import (
    DataFrame,
    SparkSession,
    functions as F,
)
from scipy.cluster.hierarchy import (
    leaves_list,
    linkage,
)

from src.util.variable_scalers import (
    VariableECDFScaler,
    VariableStandardScaler,
)
from workflow.scripts.config import Config
from workflow.scripts.constants import get_features
from workflow.scripts.data_extractor import DataExtractor
from workflow.scripts.data_processing_job import DataProcessingJob
from workflow.scripts.logger import get_logger
from workflow.scripts.plotting_functions import (
    get_fig_box,
    get_plot_patient_journey,
    get_plot_variables_distribution,
    plot_variable_distribution,
)
from workflow.scripts.spark import get_spark
from workflow.scripts.statistics_job import StatisticsJob


class DataPlottingJob:
    outputs = {
        'distributions': f'{Config.data_dir}/plots/distributions/',
        'journeys': f'{Config.data_dir}/plots/journeys/',
        'correlation_matrix': f'{Config.data_dir}/plots/correlation_matrix.pdf',
        'ecdfs': f'{Config.data_dir}/plots/ecdfs.pdf',
    }
    
    def __init__(self, spark: SparkSession, output_path: str):
        self.spark = spark
        self.data_extractor = DataExtractor(spark)
        self.data_processing_job = DataProcessingJob(self.data_extractor, output_path)
        self.show_plot = Config.remote_run
        self.logger = get_logger()
        os.makedirs(f'{Config.data_dir}/plots', exist_ok=True)
    
    def get_all_events(self) -> DataFrame:
        splits = 'train', 'val', 'test'
        all_events = []
        for split in splits:
            all_events.append(spark.read.parquet(
                self.data_processing_job.outputs[f"{split}_events"]))
        return reduce(DataFrame.unionByName, all_events)
    
    def run(self):
        events = self.data_processing_job.process_outliers.get_cached_df(self.spark)
        self.plot_correlation_matrix(events)
        self.plot_data_ecdf(events, ['FiO2', 'Albumin 5%'])
        variables_statistics = pd.read_csv(StatisticsJob.outputs['variables'])
        events = self.get_all_events()
        self.plot_patient_journeys(events, variables_statistics)
        feature_events = DataProcessingJob.process_event_values.get_cached_df(self.spark)
        feature_events = self.get_all_events()
        percentiles = (0.01, 0.99)
        variables_statistics = (
            feature_events.groupBy('variable').agg(
                *(F.expr(f'percentile(value, {p})').alias(f'p{p}') for p in percentiles),
            ))
        
        features = get_features()
        variables_df = self.spark.createDataFrame(
            pd.DataFrame(
                [f for f in features if f['name'] in {'Glucose (Blood)', 'Bilirubin (Total)'}],
                columns=['codes', 'name'])
            .rename(columns={'name': 'variable', 'codes': 'code'})
            .explode('code')
        )
        
        events_to_plot = (
            feature_events.alias('e')
            .join(variables_df, on='variable', how='inner')
            .join(variables_statistics.alias('s'), on='variable', how='inner')
            # filter out extreme outliers to make the plots more focused on the main distribution
            .filter(F.col('value').between(F.col('`p0.01`'), F.col('`p0.99`')))
            .select('e.*', 'code') # WARN: randomly selected first code will be used
        ).checkpoint()
        
        # self.plot_variables_distributions(events_to_plot, variables_df)
        self.plot_transform_comparison(events_to_plot, variables_df, 'Bilirubin (Total)')
    
    def plot_correlation_matrix(self, events):
        daily_events: pd.DataFrame = (
            events
            .withColumn('day', F.date_trunc('day', 'time').cast('long'))
            .groupBy('stay_id', 'day')
            .pivot('variable')
            .agg(F.mean('value').cast('float'))
            .drop('stay_id', 'day')
        ).toPandas()
        corr_matrix = daily_events.corr(method='spearman', min_periods=30, numeric_only=True)
        corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
        
        # select 50 most significant correlation variables
        rs_matrix = corr_matrix.abs()
        top_idx = rs_matrix.mean().sort_values(ascending=False).head(50).index
        significant_corr_matrix = rs_matrix.loc[top_idx, top_idx]
        
        # Perform hierarchical clustering
        Z = linkage(significant_corr_matrix.fillna(0), method='average')
        idx = leaves_list(Z)
        ordered_corr_matrix: pd.DataFrame = significant_corr_matrix.iloc[idx, idx]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(ordered_corr_matrix,
                    annot=False,
                    fmt=".2f",
                    cmap='viridis',
                    cbar=True,
                    square=True,
                    linewidths=0.5, ax=ax,
                    vmin=0,
                    norm=colors.LogNorm(vmin=1e-1,
                                        vmax=ordered_corr_matrix.max().max()),
                    )
        fig.tight_layout()
        fig.show()
        fig.savefig(self.outputs['correlation_matrix'],
                    bbox_inches=get_fig_box(fig))
    
    def plot_vectors(self, events):
        var = {'HR', 'MBP', "O2 Saturation", "SBP", "Temperature", "Urine"}
        ev = events \
            .filter((F.col('stay_id') == 272850) & F.col('variable').isin(var)
                    & F.col('minute').between(200, 20 * 60)) \
            .select('minute', 'variable', 'value').orderBy('minute').toPandas()
        
        ev['variable'] = ev['variable'].replace({
            'O2 Saturation': 'O2',
            "Temperature": "Temp°C",
        })
        
        fig, ax = plt.subplots(figsize=(12, 3))
        sns.heatmap(ev[['minute', 'value']].T, cmap='viridis', annot=False, cbar=False, ax=ax)
        ax.set_xticks([])
        for i, variable in enumerate(ev['variable']):
            ax.text(i + 0.5, -.5, variable, ha='center', va='center', rotation=90, fontsize=10)
        
        fig.tight_layout()
        fig.show()
        fig.savefig(f'/tmp/events_map.pdf', bbox_inches=get_fig_box(fig))
        
        # give random ids from 1 to 129 to variables
        u_vars = ev['variable'].unique()
        var_ids = np.random.choice(range(1, 130), len(u_vars), replace=False)
        
        # new df with cols 'variable_id' and mask
        # for selected variables mask is 1, for others 0
        # variables
        df = pd.DataFrame({'variable_id': range(1, 130)})
        df['mask'] = 0
        df.loc[df['variable_id'].isin(var_ids), 'mask'] = 1
        
        df = pd.DataFrame({'variable': ['HR', 'MBP', 'Insulin', 'O2', 'SBP', 'RBC', 'Temp°C',
                                        'Urine'],
                           'values': np.random.rand(8),
                           'mask': [1, 1, 0, 1, 1, 0, 1, 1]})
        
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.heatmap(df[['values']].T, cmap='viridis', annot=False, cbar=False, ax=ax)
        ax.set_xticks([])
        
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(['value', 'mask', 'variable'], rotation=0)
        
        for i, event in enumerate(df.iterrows()):
            # mask and var name
            mask, variable = event[1]['mask'], event[1]['variable']
            ax.text(i + 0.5, 1.5, mask, ha='center', va='center', fontsize=10)
            ax.text(i + 0.5, 2.5, variable, ha='center', va='center', rotation=90, fontsize=10)
        fig.tight_layout()
        fig.savefig(f'/tmp/variables_map.svg')
    
    def plot_patient_journeys(self, events: DataFrame, variables_statistics: pd.DataFrame):
        selected_journeys = {292741, 275225}
        removed_variables = {'GCS_eye', 'GCS_motor', 'GCS_verbal', 'Weight', 'INR', 'PT', 'PTT'}
        plot_journeys, schema = get_plot_patient_journey(
            self.outputs['journeys'],
            variables_statistics,
        )
        
        plots = (events
                 .withColumn('source', F.col('source').getItem(0))
                 .filter(~F.col('variable').isin(removed_variables))
                 .repartition('stay_id')
                 .filter(F.col('stay_id').isin(selected_journeys))
                 .groupBy('stay_id')
                 .applyInPandas(plot_journeys, schema=schema)).cache()
        self.logger.info(f'{plots.count()} plots saved to {self.outputs["journeys"]}')
        plots.show(100, False)
    
    def plot_variables_distributions(
        self, events: DataFrame,
        variables_df: DataFrame
    ):
        plot_dists, schema = get_plot_variables_distribution(self.outputs['distributions'])
        
        # events_sample = sample_events(events_to_plot, 10000)
        plots = (events.groupBy('variable')
                 .cogroup(variables_df.groupBy('variable'))
                 .applyInPandas(plot_dists, schema=schema)).cache()
        self.logger.info(f'{plots.count()} plots saved to {self.outputs["distributions"]}')
        plots.show(100, False)
    
    def plot_transform_comparison(self, events, variables_data: DataFrame, variable: str):
        var_data = variables_data.toPandas()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        (orig_ax, z_ax, ecdf_ax) = axes
        events_df = events \
            .filter(F.col('variable') == variable) \
            .toPandas().astype({'variable': 'category'})
        ecdf_scaler = VariableECDFScaler()
        ecdf_scaler.fit(events_df)
        
        # bins uniformly distributed between min and max
        bin_num = 16
        bin_edges = np.linspace(events_df['value'].min(), events_df['value'].max(), bin_num)
        
        plot_variable_distribution((variable,), events_df, var_data,
                                   bin_edges=bin_edges, ax=orig_ax, clip=(0, 10000), bw_adjust=0.1)
        
        transformed = ecdf_scaler.transform(events_df)
        bin_edges = np.linspace(transformed['value'].min(), transformed['value'].max(), bin_num)
        plot_variable_distribution(
            (variable,), transformed, var_data,
            bin_edges=bin_edges, ax=ecdf_ax, clip=(0, 1), bw_adjust=1.3)
        
        standard_scaler = VariableStandardScaler()
        standard_scaler.fit(events_df)
        transformed = standard_scaler.transform(events_df)
        bin_edges = np.linspace(transformed['value'].min(), transformed['value'].max(), bin_num)
        plot_variable_distribution((variable,), transformed, var_data,
                                   bin_edges=bin_edges, ax=z_ax, bw_adjust=0.1)
        
        z_ax.set_xlabel('Z-score')
        ecdf_ax.set_xlabel('quantile')
        # remove y-labels from all but the first plot
        for ax in axes[1:]:
            ax.set_ylabel('')
        # remove legends
        for ax in axes:
            ax.get_legend().remove()
        
        fig.tight_layout()
        file_name = 'transforms_' + re.sub(r"[^a-zA-Z0-9]", "_", variable) + ".pdf"
        fig.savefig(f'{self.outputs['distributions']}/{file_name}', bbox_inches=get_fig_box(fig))
        fig.show()
    
    def plot_data_ecdf(self, events: DataFrame, variables: list[str]):
        df = events \
            .filter(F.col('variable').isin(variables)) \
            .select('variable', 'value') \
            .sort('value') \
            .toPandas().astype({'variable': 'category'})
        ecdf_scaler = VariableECDFScaler()
        ecdf_scaler.fit(df)
        transformed = ecdf_scaler.transform(df)
        
        fig, axes = plt.subplots(1, len(variables), figsize=(12, 6), sharey=True)
        
        for ax, variable in zip(axes, variables):
            df_variable = df[df['variable'] == variable]
            transformed_variable = transformed[transformed['variable'] == variable]
            ax.plot(df_variable['value'], transformed_variable['value'])
            ax.set_xlabel('Original values')
            # ax.set_ylabel('P(X <= x)')
            ax.set_ylabel(r'$P(X \leq x)$')
            ax.set_title(f'ECDF function for {variable}')
        fig.tight_layout()
        fig.show()
        fig.savefig(self.outputs['ecdfs'],
                    bbox_inches=get_fig_box(fig))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()
    spark = get_spark("Plotting Job")
    job = DataPlottingJob(spark, args.output_path)
    job.run()
    spark.stop()
