import argparse
import itertools
import logging
import os
from functools import reduce
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq
from pyspark.sql import (
    DataFrame,
    SparkSession,
    functions as F,
)

from workflow.scripts.config import Config
from workflow.scripts.constants import get_features
from workflow.scripts.data_extractor import DataExtractor
from workflow.scripts.data_processing_job import DataProcessingJob
from workflow.scripts.misc import sample_events
from workflow.scripts.spark import get_spark


class StatisticsJob:
    outputs = {
        key: f"{Config.data_dir}/statistics/{key}.csv" for key in
        ('features', 'variables', 'patient_journey', 'training', 'raw_dataset', 'splits_table', 'raw_numbers')
    }
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.data_extractor = DataExtractor(spark)
        self.logger = logging.getLogger(__name__)
        out_dir = f'{Config.data_dir}/statistics'
        os.makedirs(out_dir, exist_ok=True)
    
    def run(self, output_path: str):
        """
        Compute statistics for raw and preprocessed datasets.
        Stores the results in the statistics directory as csv files.
        :return:
        """
        raw_numbers = self.raw_numbers()
        self.save_as(raw_numbers, 'raw_numbers')
        processing_outputs = DataProcessingJob(self.data_extractor, output_path).outputs
        features = get_features()
        items = self.data_extractor.read_items()
        features_df = self.compute_features_statistics(features, items)
        self.save_as(features_df, 'features')

        # training data statistics: train/test/val lengths, death prevalence
        splits = ('train', 'test', 'val')
        labels, events = {}, {}
        for split in splits:
            labels[split] = pd.read_parquet(processing_outputs[f"{split}_mortality_labels"])
            events[split] = pd.read_parquet(processing_outputs[f"{split}_events"])

        training = self.compute_splits_statistics(labels, events)
        self.save_as(training, 'training')
        splits_table = self.compute_splits_statistics_table(labels, events, icustays)
        icustays = self.data_extractor.read_icustays()
        self.save_as(splits_table, 'splits_table')
        
        data_dir = Path(Config.data_dir) / 'raw'
        # get all files and dir sizes
        data_files = data_dir.glob('*.parquet')
        raw_dataset = self.compute_raw_dataset_statistics(data_files)
        self.save_as(raw_dataset, 'raw_dataset')
        all_events = []
        for split in splits:
            all_events.append(spark.read.parquet(processing_outputs[f"{split}_events"]))
        all_events = reduce(DataFrame.unionByName, all_events)
        variables = self.compute_variables_statistics(all_events)
        self.save_as(variables, 'variables')
        patient_journey_stats = self.compute_patient_journey_statistics(all_events, icustays)
        self.save_as(patient_journey_stats, 'patient_journey')
        
        self.logger.info('Statistics computed and saved.')
    
    def save_as(self, df: pd.DataFrame, key: str):
        df.to_csv(self.outputs[key], index=False, header=True)
        self.logger.info(f'Saved {key} statistics to {self.outputs[key]}')
    
    def compute_variables_statistics(self, events: DataFrame):
        @F.udf(returnType='double')
        def wasserstein_distance_udf(mu, std, values):
            from scipy.stats import (norm)
            import numpy as np
            if std == 0:
                return None
            z_transformed = (np.array(values) - mu) / std
            data_sorted = np.sort(z_transformed)
            n = len(data_sorted)
            
            probabilities = (np.arange(1, n + 1) - 0.5) / n
            normal_quantiles = norm.ppf(probabilities)
            distance = np.mean(np.abs(data_sorted - normal_quantiles))
            return float(distance)
        
        events_sample = sample_events(events, 5000)
        norm_pvalues = (
            events_sample.groupBy('variable')
            .agg(
                F.mean('value').alias('mu'),
                F.stddev('value').alias('std'),
                F.collect_list('value').alias('values')
            )
            .withColumn('wasserstein_distance',
                        wasserstein_distance_udf(F.col('mu'), F.col('std'), F.col('values')))
            .select('variable', 'wasserstein_distance')
        )
        
        # has to be tuple to have the right brackets
        percentiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        
        variables_statistics = (
            events.groupBy('variable').agg(
                F.mean('value').alias('mean'),
                F.stddev('value').alias('std'),
                F.expr(f'percentile(value, array{percentiles})').alias('percentiles'),
                F.min('value').alias('min'),
                F.max('value').alias('max'),
                F.count('value').alias('count'),
                F.skewness('value').alias('skewness'),
                F.kurtosis('value').alias('kurtosis'),
            ).withColumn('cv', F.col('std') / F.col('mean'))
            .withColumns({f'p{p}': F.expr(f'percentiles[{i}]') for i, p in enumerate(percentiles)})
            .drop('percentiles')
            .orderBy('count', ascending=False))
        
        variables_statistics = variables_statistics.join(norm_pvalues, on='variable', how='left')
        
        return variables_statistics.toPandas()
    
    def compute_features_statistics(self, features, items: DataFrame) -> pd.DataFrame:
        all_feature_codes = list(itertools.chain.from_iterable(
            (v['codes'] for v in features)))
        all_feature_names = {v['name'] for v in features}
        num_selected_codes = len(all_feature_codes)
        assert num_selected_codes == len(set(all_feature_codes))
        
        num_categorical_features = sum(feature.get('categorical', False) for feature in features)
        
        num_raw_features = items.select('code').distinct().count()
        
        results_df = pd.DataFrame([
            ('NumberOfCategoricalFeatures', num_categorical_features),
            ('NumberOfSelectedCodes', num_selected_codes),
            ('NumberOfFeatures', len(all_feature_names)),
            ('NumberOfRawFeatures', num_raw_features),
        ], columns=['statistic', 'value'], dtype=object)
        return results_df
    
    def compute_patient_journey_statistics(self, events: DataFrame, icustays: DataFrame):
        # mean number of events per stay
        events_num = events.groupBy('stay_id').count().agg(
            F.mean('count').alias('MeanNumOfEventsPerStay'),
            F.stddev('count').alias('StdNumOfEventsPerStay'),
            F.min('count').alias('MinNumOfEventsPerStay'),
            F.max('count').alias('MaxNumOfEventsPerStay'),
        ).toPandas()
        
        # duration of stay
        durations = icustays \
            .select(((F.col('time_end') - F.col('time_start')) / 60).cast('int').alias('duration')) \
            .agg(
            F.mean('duration').alias('MeanDurationMin'),
            F.stddev('duration').alias('StdDurationMin'),
            F.min('duration').alias('MinDurationMin'),
            F.max('duration').alias('MaxDurationMin'),
        ).toPandas()
        
        # number of stays per patient
        stays_per_patients = icustays.groupBy('patient_id').count().agg(
            F.mean('count').alias('MeanNumOfStaysPerPatient'),
            F.stddev('count').alias('StdNumOfStaysPerPatient'),
            F.min('count').alias('MinNumOfStaysPerPatient'),
            F.max('count').alias('MaxNumOfStaysPerPatient'),
        ).toPandas()
        
        events_num_per_patient = events.join(icustays, on='stay_id', how='inner') \
            .groupBy('patient_id').count().agg(
            F.count('count').alias('NumPatientsWithICUStay'),
            F.mean('count').alias('MeanNumOfEventsPerPatient'),
            F.stddev('count').alias('StdNumOfEventsPerPatient'),
            F.min('count').alias('MinNumOfEventsPerPatient'),
            F.max('count').alias('MaxNumOfEventsPerPatient'),
        ).toPandas()
        
        result = pd.concat(
            [events_num, durations, stays_per_patients, events_num_per_patient],
            axis=1).astype(object).T.reset_index()
        result.columns = ['statistic', 'value']
        return result
    
    def compute_splits_statistics_table(
        self,
        labeled_dfs: dict[str, pd.DataFrame],
        event_dfs: dict[str, pd.DataFrame],
        icustays: DataFrame
    ):
        # Convert 'icustays' Spark DataFrame to Pandas DataFrame
        icu_patient_ids = icustays.select('patient_id', 'stay_id').distinct().toPandas()
        
        # Initialize a list to collect rows of data
        rows = []
        
        # Process supervised splits
        for split, labels in labeled_dfs.items():
            split_events = event_dfs[split]
            if split == 'train':
                # count only labeled
                num_events = split_events['stay_id'].isin(labels['stay_id']).sum()
                split_name = 'supervised_train'
            else:
                num_events = split_events.shape[0]
                split_name = split
            num_patients = icu_patient_ids.loc[
                icu_patient_ids['stay_id'].isin(labels['stay_id']), 'patient_id'
            ].nunique()
            num_stays = labels['stay_id'].nunique()
            rows.append({
                'split': split_name,
                'num_events': num_events,
                'num_patients': num_patients,
                'num_stays': num_stays,
            })
        
        # Process unsupervised training data
        unsupervised_events = event_dfs['train']
        num_events = unsupervised_events.shape[0]
        num_patients = icu_patient_ids.loc[
            icu_patient_ids['stay_id'].isin(unsupervised_events['stay_id']), 'patient_id'
        ].nunique()
        num_stays = icu_patient_ids.loc[
            icu_patient_ids['stay_id'].isin(unsupervised_events['stay_id']), 'stay_id'
        ].nunique()
        rows.append({
            'split': 'unsupervised_train',
            'num_events': num_events,
            'num_patients': num_patients,
            'num_stays': num_stays,
        })
        
        # Create DataFrame from collected rows
        result = pd.DataFrame(rows)
        
        return result
    
    def compute_splits_statistics(
        self,
        labels: dict[str, pd.DataFrame],
        events: dict[str, pd.DataFrame]
    ):
        data = []
        for split, df in labels.items():
            data.append((f"Supervised{split.capitalize()}DeathPrevalence", df['died'].mean()))
            data.append((f"Supervised{split.capitalize()}LengthLabels", df.shape[0]))
            num_events = events[split]['stay_id'].isin(df['stay_id']).sum()
            data.append((f"Supervised{split.capitalize()}LengthEvents", num_events))
        
        for split, df in events.items():
            data.append((f"Unsupervised{split.capitalize()}LengthEvents", df.shape[0]))
            # TODO: number of stays per split
        
        return pd.DataFrame(data, columns=['statistic', 'value'], dtype=object)
    
    def compute_raw_dataset_statistics(self, parquets: Iterable[Path]):
        def get_row(parquet_file: Path):
            dataset = pq.ParquetDataset(parquet_file)
            num_rows = sum(f.metadata.num_rows for f in dataset.fragments)
            file_size = sum(Path(f).stat().st_size / 2 ** 20 for f in dataset.files)
            return parquet_file.name, num_rows, file_size
        
        rows = [get_row(p) for p in parquets]
        df = pd.DataFrame(rows, columns=['file', 'rows', 'size_mb'])
        return df
    
    def single_variable_stat(self, variable_name):
        events = self.spark.read.parquet(processing_outputs['unsupervised_splits'])
        var_events = events.filter(F.col('variable') == variable_name)
        var_events.groupBy('source').count().show()
        
        var_stat = var_events.agg(
            F.mean('value').alias('mean'),
            F.stddev('value').alias('std'),
            F.min('value').alias('min'),
            F.max('value').alias('max'),
            F.count('value').alias('count'),
        )
        var_stat.show()
        
        var_events.groupBy('source').agg(
            F.mean('value').alias('mean'),
            F.stddev('value').alias('std'),
            F.min('value').alias('min'),
            F.max('value').alias('max'),
            F.count('value').alias('count'),
        ).show()
    
    def raw_numbers(self):
        """
        Get numbers from raw unprocessed dataset
        :return:
        """
        total_patients = self.data_extractor.read_patients()
        total_icustays = self.data_extractor.read_icustays()
        
        df = pd.DataFrame([
            ('TotalPatients', total_patients.count()),
            ('TotalICUStays', total_icustays.count()),
        ], columns=['statistic', 'value'], dtype=object)
        return df
        
    
    def count_sanitized_events(self):
        events = DataProcessingJob.process_outliers.get_cached_df(self.spark)
        print(f"Num sanitized events")
        events.groupBy('source').count().show()
    
    def hourly_inputevents_mv(self):
        dpj = DataProcessingJob(self.data_extractor)
        inputevents_mv = self.data_extractor.read_inputevents_mv()
        total_count = inputevents_mv.count()
        print(f"Total events: {total_count}")
        
        hour_plus_count = inputevents_mv.filter(
            (F.col('time_end') - F.col('time_start')).cast('int') > 3600).count()
        print(f"Num events longer than 1h: {hour_plus_count}")
        
        inputevents_mv = dpj.distribute_events_hourly(inputevents_mv)
        hourly_count = inputevents_mv.count()
        print(hourly_count)
    
    def icustays(self):
        icustays = self.data_extractor.read_icustays()
        print(f"Num icustays {icustays.count()}")
    
    def labevents(self):
        labevents = self.data_extractor.read_labevents()
        print(f"Num raw labevents {labevents.count()}")
        
        sanitized_labevents = DataProcessingJob.process_outliers.get_cached_df(self.spark)
        sanitized_labevents = sanitized_labevents.filter(F.col('source') == 'labevents')
        sanitized_labevents.groupBy('variable').count().show(1000)
        print(f"Num sanitized labevents {sanitized_labevents.count()}")
    
    def chartevents(self):
        chartevents = self.data_extractor.read_chartevents()
        print(f"Num raw chartevents {chartevents.count()}")
        
        sanitized_chartevents = DataProcessingJob.process_outliers.get_cached_df(self.spark)
        sanitized_chartevents = sanitized_chartevents.filter(F.col('source') == 'chartevents')
        sanitized_chartevents.groupBy('variable').count().show(1000)
        print(f"Num sanitized chartevents {sanitized_chartevents.count()}")


if __name__ == '__main__':
    # spark = SparkSession.builder.appName('StatisticsJob').getOrCreate()
    
    parser = argparse.ArgumentParser()
    # FIXME: rename to --dataset-path
    parser.add_argument('--output-path', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()
    
    spark = get_spark('StatisticsJob')
    job = StatisticsJob(spark)
    job.run(args.output_path)
    # job.chartevents()
    # job.labevents()
    # job.count_sanitized_events()
    # job.single_variable_stat('D5W')
    # job.icustays()
    # job.hourly_inputevents_mv()
    spark.stop()
