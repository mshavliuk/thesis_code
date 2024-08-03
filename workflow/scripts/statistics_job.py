import itertools
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
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


class StatisticsJob:
    outputs = {key: f"{Config.data_dir}/statistics/{key}.csv" for key in
               ('features', 'variables', 'patient_journey', 'training', 'raw_dataset')}
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.data_extractor = DataExtractor(spark)
        self.logger = logging.getLogger(__name__)
        out_dir = f'{Config.data_dir}/statistics'
        os.makedirs(out_dir, exist_ok=True)
    
    def run(self):
        """
        Compute statistics for raw and preprocessed datasets.
        Stores the results in the statistics directory as csv files.
        :return:
        """
        
        features = get_features()
        items = self.data_extractor.read_items()
        features_df = self.compute_features_statistics(features, items)
        self.save_as(features_df, 'features')
        
        # training data statistics: train/test/val lengths, death prevalence
        train, test, val = (np.load(
            DataProcessingJob.outputs[f"{split}_stay_ids"], allow_pickle=True) for split in
            ('train', 'test', 'val'))
        
        labels = pd.read_pickle(DataProcessingJob.outputs['mortality_labels'])
        training = self.compute_training_statistics(train, test, val, labels)
        self.save_as(training, 'training')
        
        data_dir = Path(Config.data_dir) / 'raw'
        # get all files and dir sizes
        data_files = data_dir.glob('*.parquet')
        raw_dataset = self.compute_raw_dataset_statistics(data_files)
        self.save_as(raw_dataset, 'raw_dataset')
        
        events = self.spark.read.parquet(DataProcessingJob.outputs['events'])
        variables = self.compute_variables_statistics(events)
        self.save_as(variables, 'variables')
        
        icustays = self.data_extractor.read_icustays()
        patient_journey_stats = self.compute_patient_journey_statistics(events, icustays)
        self.save_as(patient_journey_stats, 'patient_journey')
        
        self.logger.info('Statistics computed and saved.')
    
    def save_as(self, df: pd.DataFrame, key: str):
        df.to_csv(self.outputs[key], index=False, header=True)
    
    def compute_variables_statistics(self, events: DataFrame):
        # has to be tuple to have the right brackets
        percentiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        # TODO: compute statistics for interval between consequent measurements of the same variable
        
        variables_statistics = (
            events.groupBy('variable').agg(
                F.mean('value').alias('mean'),
                F.stddev('value').alias('std'),
                F.expr(f'percentile_approx(value, array{percentiles})').alias('percentiles'),
                F.min('value').alias('min'),
                F.max('value').alias('max'),
                F.count('value').alias('count'),
            ).withColumn('cv', F.col('std') / F.col('mean'))
            .withColumns({f'p{p}': F.expr(f'percentiles[{i}]') for i, p in enumerate(percentiles)})
            .drop('percentiles')
            .orderBy('count', ascending=False))
        
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
    
    def compute_training_statistics(self, train, test, val, labels):
        death_prevalence = labels['died'].mean()
        
        return pd.DataFrame([
            ('TrainSplitLength', len(train)),
            ('TestSplitLength', len(test)),
            ('ValSplitLength', len(val)),
            ('DeathPrevalence', death_prevalence),
        ], columns=['statistic', 'value'], dtype=object)
    
    def compute_raw_dataset_statistics(self, parquets: Iterable[Path]):
        def get_row(parquet_file: Path):
            dataset = pq.ParquetDataset(parquet_file)
            num_rows = sum(f.metadata.num_rows for f in dataset.fragments)
            file_size = sum(Path(f).stat().st_size / 2 ** 20 for f in dataset.files)
            for fragment in dataset.fragments:
                num_rows += fragment.metadata.num_rows
            return parquet_file.name, num_rows, file_size
        
        rows = [get_row(p) for p in parquets]
        df = pd.DataFrame(rows, columns=['file', 'rows', 'size_mb'])
        return df
