import argparse
import itertools
import logging
import os
import pickle
from functools import (
    reduce,
)
from typing import TypedDict

import pandas as pd
from pyspark.sql import (
    DataFrame,
    functions as F,
)
from pyspark.sql.functions import udf
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StructField,
    StructType,
    TimestampType,
)

from workflow.scripts.cache_result import (
    cache_result,
)
from workflow.scripts.constants import get_features
from workflow.scripts.data_extractor import DataExtractor
from workflow.scripts.spark import get_spark

class Splits[T](TypedDict):
    train: T
    val: T
    test: T

class DataProcessingJob:
    def __init__(self, data_extractor: DataExtractor, output_path: str):
        self.data_extractor = data_extractor
        self.logger = logging.getLogger(__name__)
        
        self.outputs = {
            'train_mortality_labels': f'{output_path}/train/mortality_labels.parquet',
            'test_mortality_labels': f'{output_path}/test/mortality_labels.parquet',
            'val_mortality_labels': f'{output_path}/val/mortality_labels.parquet',
            'train_events': f'{output_path}/train/events.parquet',
            'test_events': f'{output_path}/test/events.parquet',
            'val_events': f'{output_path}/val/events.parquet',
            'train_demographics': f'{output_path}/train/demographics.parquet',
            'test_demographics': f'{output_path}/test/demographics.parquet',
            'val_demographics': f'{output_path}/val/demographics.parquet',
            'unittest_events': './src/util/tests/data/events.parquet',
            'unittest_mortality_labels': './src/util/tests/data/mortality_labels.parquet',
            'unittest_demographics': './src/util/tests/data/demographics.parquet',
        }
        
        out_dirs = {os.path.abspath(os.path.dirname(out)) for out in self.outputs.values()}
        
        for out_dir in out_dirs:
            os.makedirs(out_dir, exist_ok=True)
    
    @staticmethod
    @udf(ArrayType(StructType([
        StructField('time', TimestampType(), False),
        StructField('value', FloatType(), False),
    ])))
    def distribute_value_in_time_udf(amount, time, num_hours):
        # FIXME: num_hours could be negative sometimes
        if num_hours <= 1:
            yield time, amount
            return
        
        hour_amount = amount / num_hours
        for i in range(int(num_hours)):
            yield time - pd.Timedelta(hours=num_hours - i - 1), hour_amount
        
        if (num_hours % 1) > 0:
            # FIXME: `num_hours % 1` sometimes produces tiny fractions of amounts and creates extra
            #  23560 residual entries. Instead, if num_hours % 1 < 1/60 it's better to split the amount
            #  into num_hours equal parts
            yield time, hour_amount * (num_hours % 1)
    
    def distribute_events_hourly(self, events) -> DataFrame:
        # TODO: check if it gets faster if we split the events into smaller partitions
        # Get number of workers
        num_workers = self.data_extractor.spark.sparkContext.defaultParallelism
        events = events.repartition(num_workers).checkpoint(eager=False)
        hourly_events = (
            events
            .withColumn('duration', (F.col('time_end') - F.col('time_start')).cast('long') / 3600)
            .withColumn('hourly_amounts', DataProcessingJob.distribute_value_in_time_udf(
                F.col('value'), F.col('time_end'), F.col('duration')))
            .withColumn('exploded', F.explode('hourly_amounts'))
            .withColumns({
                # coalesce is used to make schema not nullable, but otherwise have no effect
                'time': F.coalesce(F.col('exploded.time'), F.col('time_end')),
                'value': F.coalesce(F.col('exploded.value'), F.col('value')),
            })
            .drop('hourly_amounts', 'exploded')
        )
        return hourly_events
    
    def convert_units(self, events):
        # TODO: see what this returns and check if units are compatible
        return (events.withColumn(
            'value',
            F.when(F.col('unit') == 'mcg', F.col('value') * 0.001)
            # .when(F.col('unit').startswith('gram') & F.col('unit') == 'gm', F.col('value') * 1000)
            .when(F.col('unit') == 'L', F.col('value') * 1000)
            .otherwise(F.col('value')))
        .withColumn(
            'unit',
            F.when(F.col('unit') == 'mcg', 'mg')
            # .when(F.col('unit').startswith('gram') & F.col('unit') == 'gm', 'mg')
            .when(F.col('unit') == 'L', 'ml')
            .otherwise(F.col('unit'))))
    
    @cache_result(f'raw_events', partitionBy='code')
    def get_raw_events(self, features) -> DataFrame:
        chartevents = (self.data_extractor.read_chartevents()
                       .withColumn('source', F.lit('chartevents')))
        
        labevents = self.data_extractor.read_labevents()
        labevents = self.add_stay_id_col(labevents, self.data_extractor.read_icustays()) \
            .withColumn('source', F.lit('labevents'))
        
        outputevents = (self.data_extractor.read_outputevents()
                        .withColumn('source', F.lit('outputevents')))
        weight_events = (self.data_extractor.read_weight_events()
                         .withColumn('source', F.lit('weight')))
        # TODO: check if cv events also need to be distributed in time
        inputevents_cv = (self.data_extractor.read_inputevents_cv()
                          .withColumn('source', F.lit('inputevents_cv')))
        inputevents_mv = self.data_extractor.read_inputevents_mv()
        inputevents_mv = (self.distribute_events_hourly(inputevents_mv)
                          .withColumn('source', F.lit('inputevents_mv')))
        
        combined_inputevents = self.convert_units(
            inputevents_cv.unionByName(inputevents_mv, allowMissingColumns=True))
        
        all_feature_codes = list(itertools.chain.from_iterable(
            (v['codes'] for v in features)))
        feature_codes_df = self.data_extractor.spark \
            .createDataFrame(all_feature_codes, IntegerType()).toDF('code')
        
        raw_events = (
            chartevents
            .unionByName(labevents, allowMissingColumns=True)
            .unionByName(outputevents, allowMissingColumns=True)
            .unionByName(combined_inputevents, allowMissingColumns=True)
            .unionByName(weight_events, allowMissingColumns=True)
            # filter events for selected features
            .join(feature_codes_df, on='code', how='inner')
            .select('value', 'code', 'unit', 'time', 'source', 'stay_id'))
        return raw_events
    
    def apply_feature_selectors(self, events: DataFrame, features):
        """
        Process all events by selecting the target column (either VALUE or VALUENUM) as given by the feature dict
        :param events:
        :param features:
        :return:
        """
        value_coalesces = []
        
        for feature in features:
            if 'select' in feature:
                value_coalesces.append(
                    F.when(F.col('code').isin(feature['codes']), feature['select'])
                )
        value_coalesces.append(F.col('value'))
        
        feature_events = events.withColumn(
            'feature_value', F.coalesce(*value_coalesces).cast(FloatType()),
        )
        return feature_events
    
    @cache_result('feature_events', partitionBy='variable')
    def process_event_values(self, events: DataFrame, features) -> DataFrame:
        events = events.repartition('code')
        feature_events = (
            self.apply_feature_selectors(events, features)
            .filter(F.col('feature_value').isNotNull())
            .drop('value')
            .withColumnRenamed('feature_value', 'value'))
        feature_events = self.add_variable_name_col(feature_events, features)
        return feature_events
    
    @cache_result('sanitized_events', partitionBy='variable')
    def process_outliers(self, events: DataFrame, features) -> DataFrame:
        filter_configs = {}
        medians = []
        
        for feature in features:
            if (config := feature.get('filtering', None)) is not None:
                if feature['name'] not in filter_configs:
                    filter_configs[feature['name']] = config
                    if config['action'] == 'replace_with_median':
                        median = events \
                            .filter((F.col('variable') == feature['name']) & config['valid']) \
                            .agg(F.expr('percentile_approx(value, 0.5)').alias('median')) \
                            .withColumn('variable', F.lit(feature['name']))
                        medians.append(median)
                elif str(filter_configs[feature['name']]) != str(config):
                    raise ValueError(f'Filtering configs for variable {feature["name"]} are different')
            else:
                filter_configs[feature['name']] = {}  # no filtering
        
        events = events.repartition('variable')
        variables = []
        medians_df = reduce(DataFrame.unionByName, medians).cache()
        
        for variable, config in filter_configs.items():
            feature_events = events.filter(F.col('variable') == variable)
            if not config:  # no filtering, append as is
                variables.append(feature_events)
            elif config['action'] == 'remove':
                feature_events = (feature_events.filter(config['valid']))
                variables.append(feature_events)
            elif config['action'] == 'replace_with_median':
                feature_events = feature_events \
                    .join(medians_df, on='variable', how='left') \
                    .withColumn(
                    'value',
                    F.when(config['valid'], F.col('value')).otherwise(F.col('median'))
                ).drop('median')
                variables.append(feature_events)
            else:
                raise ValueError(f'Unknown outliers value {config["outliers"]}')
        
        # unite all variables
        events = reduce(DataFrame.unionByName, variables)
        return events
    
    def get_demographics(self, icu_stays) -> DataFrame:
        ages = self.extract_ages(icu_stays)
        patients = self.data_extractor.read_patients()
        genders = self.extract_gender_events(icu_stays, patients)
        # TODO: add ethnicity
        
        return ages.join(genders, on='stay_id', how='outer')
    
    def add_variable_name_col(self, events: DataFrame, features):
        variable_names_df = events.sparkSession.createDataFrame(
            pd.DataFrame(features, columns=['codes', 'name'])
        ).select(F.explode('codes').alias('code'), F.col('name').alias('variable'))
        events = events.join(variable_names_df, on='code', how='inner')
        return events
    
    def add_stay_id_col(self, events: DataFrame, icu_stays: DataFrame):
        if 'stay_id' in events.columns:
            raise ValueError('stay_id column already exists in events DataFrame')
        
        with_computed_icu = (
            events.alias('event')
            # inner join => all events have icu stay or filtered out
            .join(icu_stays, on='admission_id', how='inner')
            # FIXME: better to increase the time range by a few hours from both sides
            .filter(F.col('time').between(F.col('time_start'), F.col('time_end')))
            .select('event.*', 'stay_id'))
        return with_computed_icu
    
    def filter_icu_stays(self, events: DataFrame, icu_stays: DataFrame):
        # TODO: select stays where count is at least 100
        return events.groupBy('stay_id').count() \
            .filter(F.col('count') > 1) \
            .select('stay_id') \
            .join(icu_stays, on='stay_id', how='inner')
    
    def use_relative_icu_time(self, events: DataFrame, icu_stays: DataFrame):
        # TODO: shift so 0 is smallest time in stay (avoid negative minutes)
        with_rel_time = (
            events.alias('event')
            .join(icu_stays, on='stay_id', how='inner')
            .withColumn('minute', F.round(
                (F.col('time') - F.col('time_start')).cast('int') / 60
            ).cast('int'))
            .drop('time')
            .select('event.*', 'minute'))
        return with_rel_time
    
    def get_icu_alive_for_24h(self, icu_stays: DataFrame, events: DataFrame, admissions: DataFrame):
        # stay at least 24h
        icu_24_plus = (icu_stays.filter(
            ((F.col('time_end') - F.col('time_start')).cast('int') >= 24 * 60 * 60))
        )
        
        # alive for at least 24h
        icu_24_plus = (
            icu_24_plus.alias('stay')
            .join(admissions, on='admission_id', how='inner')
            .filter(F.col('death_time').isNull() | (
                (F.col('death_time') - F.col('time_start')).cast('int') >= 24 * 60 * 60))
            .select('stay.*', 'died')
        )
        # has at least one event in the first 24h
        stay_ids_with_events = (
            events
            .join(icu_24_plus, on='stay_id', how='inner')
            .filter(F.col('minute') < 24 * 60)
            .select('stay_id')
            .distinct())
        icu_24_plus = icu_24_plus.join(stay_ids_with_events, on='stay_id', how='inner')
        return icu_24_plus
    
    def extract_ages(self, icu_stays) -> DataFrame:
        ages = icu_stays.select(
            'stay_id',
            F.when(F.col('age') > 200, 91.4).otherwise(F.col('age')).alias('age'),
        )
        return ages
    
    def extract_gender_events(self, icu_stays: DataFrame, patients: DataFrame):
        gender_events = icu_stays.join(patients, on='patient_id', how='inner').select(
            'stay_id',
            F.when(F.col('GENDER') == 'M', 0).otherwise(1).alias('gender'),
        )
        return gender_events
    
    def run(self, seed: int, strats_splits_path) -> DataFrame:
        features = get_features()
        self.logger.info(f'Reading data')
        raw_events = self.get_raw_events(features)
        self.logger.info('Selecting variables')
        feature_events = self.process_event_values(raw_events, features)
        self.logger.info('Processing outliers')
        sanitized_events = self.process_outliers(feature_events, features)
        # filter ICU stays so we don't generate useless gender and age events
        icu_stays = self.data_extractor.read_icustays()
        admissions = self.data_extractor.read_admissions()
        icu_events = self.use_relative_icu_time(sanitized_events, icu_stays)
        
        # Saving data for supervised task
        icu_stays_24h = self.get_icu_alive_for_24h(icu_stays, icu_events, admissions)
        
        if strats_splits_path is not None:
            split_stay_ids = self.split_stays_by_strats(strats_splits_path)
        else:
            split_fractions: Splits[float] = {'train': .64, 'test': .2, 'val': .16 }
            split_stay_ids = self.split_stays_by_patient(icu_stays_24h, split_fractions, seed)
        self.save_label_splits(icu_stays_24h, split_stay_ids)
        
        all_events = self.throttle_events(icu_events)
        
        # Saving data for supervised + unsupervised tasks (for all icu_stays, not only 24h+)
        # Train events are all events from all stays (not only 24h+) except val+test splits
        train_icu_stay_ids = all_events.select('stay_id') \
            .subtract(split_stay_ids['test']) \
            .subtract(split_stay_ids['val'])
        
        all_split_stay_ids: Splits[DataFrame] = {
            'train': train_icu_stay_ids,
            'test': split_stay_ids['test'],
            'val': split_stay_ids['val'],
        }
        
        demographics = self.get_demographics(icu_stays)
        
        self.save_demographic_splits(demographics, all_split_stay_ids)
        self.save_event_splits(all_events, all_split_stay_ids)
        return all_events
    
    def save_label_splits(self, labels, splits: Splits[DataFrame]):
        for name, df in splits.items():
            labels.join(df, on='stay_id', how='inner') \
                .select('stay_id', 'died') \
                .toPandas().to_parquet(self.outputs[f'{name}_mortality_labels'])
    
    def save_demographic_splits(self, demographics: DataFrame, splits: Splits[DataFrame]):
        for name, df in splits.items():
            demographics.join(df, on='stay_id', how='inner').toPandas().to_parquet(
                self.outputs[f'{name}_demographics'])
    
    def save_event_splits(self, events: DataFrame, splits: Splits[DataFrame]):
        for name, df in splits.items():
            events.join(df, on='stay_id', how='inner').repartition(1).write.parquet(
                self.outputs[f'{name}_events'], mode='overwrite')
    
    def split_stays_by_strats(self, split_path) -> Splits[DataFrame]:
        data, oc, train_ids, val_ids, test_ids = pickle.load(open(split_path, 'rb'))
        
        return {
            'train': self.data_extractor.spark.createDataFrame(train_ids, schema=['stay_id']),
            'val': self.data_extractor.spark.createDataFrame(val_ids, schema=['stay_id']),
            'test': self.data_extractor.spark.createDataFrame(test_ids, schema=['stay_id']),
        }
    
    def split_stays_by_patient(
        self, icu_stays_24h: DataFrame, fractions: Splits[float], seed: int
    ) -> Splits[DataFrame]:
        assert sum(fractions.values()) == 1, 'Fractions must sum to 1'
        icu_stays_24h = icu_stays_24h.cache()
        # split events on patient level
        patient_ids = icu_stays_24h.select('patient_id').distinct().cache()
        patient_id_splits = patient_ids \
            .orderBy('patient_id') \
            .randomSplit(list(fractions.values()), seed=seed)
        # check if there are intersections
        
        result = {}
        # convert patients to stays
        for key, df in zip(fractions.keys(), patient_id_splits):
            split = icu_stays_24h.join(df, on='patient_id', how='inner').select('stay_id').cache()
            result[key] = split
        return result
    
    @cache_result('all_events')
    def throttle_events(self, events) -> DataFrame:
        events = events.groupBy('stay_id', 'minute', 'variable').agg(
            F.mean('value').alias('value'),
            F.collect_set('source').alias('source'),
            F.first('unit').alias('unit'),
        )
        
        return events
    
    def generate_unittest_dataset(self):
        demographics = self.data_extractor.spark.read.parquet(self.outputs['test_demographics'])
        events = self.data_extractor.spark.read.parquet(self.outputs['test_events'])
        labels = self.data_extractor.spark.read.parquet(self.outputs['test_mortality_labels'])
        
        stay_ids = demographics.select('stay_id').sample(0.005).cache()
        self.save_demographic_splits(demographics, {'unittest': stay_ids})
        self.save_event_splits(events, {'unittest': stay_ids})
        self.save_label_splits(labels, {'unittest': stay_ids})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-path', type=str, required=True, help='Path to output directory')
    parser.add_argument('--split-by-strats', type=str, help='Path to strats splits')
    args = parser.parse_args()
    
    spark = get_spark("Preprocessing MIMIC-III dataset")
    data_extractor = DataExtractor(spark)
    job = DataProcessingJob(data_extractor, args.output_path)
    job.run(args.seed, args.split_by_strats)
    # job.generate_unittest_dataset()
    spark.stop()
