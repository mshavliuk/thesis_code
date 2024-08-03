import os

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
from workflow.scripts.plotting_functions import (
    get_plot_patient_journey,
    get_plot_variables_distribution,
)
from workflow.scripts.statistics_job import StatisticsJob


class PlottingJob:
    outputs = {
        'distributions': f'{Config.data_dir}/plots/distributions/',
        'journeys': f'{Config.data_dir}/plots/journeys/',
        'correlation_matrix': f'{Config.data_dir}/plots/correlation_matrix.eps',
    }
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.data_extractor = DataExtractor(spark)
        self.show_plot = Config.remote_run
        self.logger = Config.get_logger(__name__)
        os.makedirs(f'{Config.data_dir}/plots', exist_ok=True)
    
    def run(self):
        # events = DataProcessingJob.sanitize_events.get_cached_df(self.spark)
        # self.plot_correlation_matrix(events)
        variables_statistics = pd.read_csv(StatisticsJob.outputs['variables'])
        events = self.spark.read.parquet(DataProcessingJob.outputs['events'])
        self.plot_patient_journeys(events, variables_statistics)
        
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
        selected_journeys = {272850, 275225, 292741}
        removed_variables = {'GCS_eye', 'GCS_motor', 'GCS_verbal', 'Weight'}
        plot_journeys, schema = get_plot_patient_journey(
            self.outputs['journeys'],
            variables_statistics,
            file_format='eps',
        )
        plots = (events
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
