import os
from functools import reduce

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import (
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

from workflow.scripts.config import Config
from workflow.scripts.constants import get_features
from workflow.scripts.data_extractor import DataExtractor
from workflow.scripts.data_processing_job import DataProcessingJob
from workflow.scripts.logger import get_logger
from workflow.scripts.plotting_functions import (
    get_plot_patient_journey,
    get_plot_variables_distribution,
)
from workflow.scripts.spark import get_spark
from workflow.scripts.statistics_job import StatisticsJob


class DataPlottingJob:
    outputs = {
        'distributions': f'{Config.data_dir}/plots/distributions/',
        'journeys': f'{Config.data_dir}/plots/journeys/',
        'correlation_matrix': f'{Config.data_dir}/plots/correlation_matrix.eps',
    }
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.data_extractor = DataExtractor(spark)
        self.show_plot = Config.remote_run
        self.logger = get_logger()
        os.makedirs(f'{Config.data_dir}/plots', exist_ok=True)
    
    def get_all_events(self) -> DataFrame:
        splits= 'train', 'val', 'test'
        all_events = []
        for split in splits:
            all_events.append(spark.read.parquet(DataProcessingJob.outputs[f"{split}_events"]))
        return reduce(DataFrame.unionByName, all_events)
    
    def run(self):
        # events = DataProcessingJob.sanitize_events.get_cached_df(self.spark)
        # self.plot_correlation_matrix(events)
        variables_statistics = pd.read_csv(StatisticsJob.outputs['variables'])
        events = self.get_all_events()
        self.plot_patient_journeys(events, variables_statistics)
        exit()
        items = self.data_extractor.read_items()
        feature_events = DataProcessingJob.process_event_values.get_cached_df(self.spark)
        
        features = get_features()
        variables_df = self.spark.createDataFrame(
            pd.DataFrame(
                [f for f in features if f['name'] in {'Glucose (Blood)', 'Temperature'}],
                columns=['codes', 'name'])
            .rename(columns={'name': 'variable', 'codes': 'code'})
            .explode('code')
        )
        statistics_df= self.spark.createDataFrame(variables_statistics)
        self.plot_variables_distributions(
            feature_events, items, statistics_df, variables_df)
    
    def plot_correlation_matrix(self, events):
        daily_events: pd.DataFrame = (
            events
            .withColumn('day', F.date_trunc('day', 'time').cast('long'))
            .groupBy('admission_id', 'day')
            .pivot('variable')
            .agg(F.mean('value').cast('float'))
            .drop('admission_id', 'day')
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
                    # norm=colors.LogNorm(vmin=0.3,
                    #                     vmax=ordered_corr_matrix.max().max()),
                    )
        fig.tight_layout()
        fig.savefig(self.outputs['correlation_matrix'])
        fig.show()
    
    def plot_patient_journeys(self, events: DataFrame, variables_statistics: pd.DataFrame):
        
        var = {'HR', 'MBP', "O2 Saturation", "SBP", "Temperature", "Urine"}
        ev = events \
            .filter((F.col('stay_id') == 272850) & F.col('variable').isin(var)
                    & F.col('minute').between(200, 20 * 60))\
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
        fig.savefig(f'/tmp/events_map.svg')
        
        
        # give random ids from 1 to 129 to variables
        u_vars = ev['variable'].unique()
        var_ids = np.random.choice(range(1, 130), len(u_vars), replace=False)
        
        # new df with cols 'variable_id' and mask
        # for selected variables mask is 1, for others 0
        # variables
        df = pd.DataFrame({'variable_id': range(1, 130)})
        df['mask'] = 0
        df.loc[df['variable_id'].isin(var_ids), 'mask'] = 1
        
        df = pd.DataFrame({'variable': ['HR', 'MBP', 'Insulin', 'O2', 'SBP', 'RBC', 'Temp°C', 'Urine'],
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
        exit()
        
        selected_journeys = {272850, 275225, 292741, 271806}
        removed_variables = {'GCS_eye', 'GCS_motor', 'GCS_verbal', 'Weight', 'INR', 'PT', 'PTT'}
        plot_journeys, schema = get_plot_patient_journey(
            self.outputs['journeys'],
            variables_statistics,
            file_formats=['png', 'eps', 'svg'],
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
        self, events: DataFrame, items: DataFrame,
        statistics_df: DataFrame, variables_df: DataFrame
    ):
        events_to_plot = (
            events.alias('e')
            .join(variables_df, on='code', how='inner')
            .join(statistics_df, on='variable', how='inner')
            # filter out extreme outliers to make the plots more focused on the main distribution
            .filter(F.col('value').between(F.col('`p0.01`'), F.col('`p0.99`')))
            .select('e.variable', 'e.value', 'e.code', 'e.unit')
        )
        
        variables_data = (
            variables_df
            # filter out variables in case a subset of events was given
            .join(events_to_plot.select('code').distinct(), on='code', how='inner')
        )
        plot_dists, schema = get_plot_variables_distribution(self.outputs['distributions'])
        events_sample = self.sample_events(events_to_plot, 10000)
        plots = (events_sample.groupBy('variable')
                 .cogroup(variables_data.groupBy('variable'))
                 .applyInPandas(plot_dists, schema=schema)).cache()
        self.logger.info(f'{plots.count()} plots saved to {self.outputs["distributions"]}')
        plots.show(100, False)
    
    def sample_events(self, events: DataFrame, sample_count: int):
        fractions = (
            events.groupby('variable')
            .count()
            .withColumn('fraction', F.least(F.lit(1), F.lit(sample_count) / F.col('count')))
            .drop('count')
            .rdd.collectAsMap()
        )
        return events.sampleBy('variable', fractions=fractions)
    
if __name__ == '__main__':
    spark = get_spark("Plotting Job")
    job = DataPlottingJob(spark)
    job.run()
    spark.stop()
