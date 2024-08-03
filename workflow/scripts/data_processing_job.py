import itertools
import logging
from functools import (
    reduce,
)

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
from workflow.scripts.config import Config
from workflow.scripts.constants import get_features
from workflow.scripts.data_extractor import DataExtractor
from workflow.scripts.spark import get_spark


class DataProcessingJob:
    outputs = {
        'mortality_labels': f'{Config.data_dir}/preprocessed/mortality_labels.pkl',
        'events': f'{Config.data_dir}/preprocessed/events.parquet',
        'train_stay_ids': f'{Config.data_dir}/preprocessed/train_stay_ids.npy',
        'test_stay_ids': f'{Config.data_dir}/preprocessed/test_stay_ids.npy',
        'val_stay_ids': f'{Config.data_dir}/preprocessed/val_stay_ids.npy',
    }
    
    def __init__(self, data_extractor: DataExtractor):
        self.data_extractor = data_extractor
        self.logger = logging.getLogger(__name__)
    
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
            .select('value', 'code', 'unit', 'admission_id', 'time', 'source', 'stay_id'))
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
    
    def process_outliers(self, events: DataFrame, features):
        filter_configs = {}
        config_keys = {'min', 'max', 'outliers'}
        for feature in features:
            if set(feature.keys()) > config_keys:
                feature_config = {k: feature[k] for k in config_keys}
                if feature['name'] not in filter_configs:
                    filter_configs[feature['name']] = feature_config
                elif filter_configs[feature['name']] != feature_config:
                    raise ValueError(f'Filtering configs for variable {feature["name"]} are different')
            elif set(feature.keys()) & config_keys:
                raise ValueError(f'Filtering config for variable {feature["name"]} should contain all (or none) of {config_keys}')
            else:
                filter_configs[feature['name']] = {}
        
        events = events.repartition('variable')
        variables = []
        medians = events.groupBy('variable').agg(F.expr('percentile_approx(value, 0.5)').alias('median'))
        
        for variable, config in filter_configs.items():
            feature_events = events.filter(F.col('variable') == variable)
            if not config:
                variables.append(feature_events)
            elif config['outliers'] == 'remove':
                feature_events = (feature_events
                                  .filter(F.col('value').between(config['min'], config['max'])))
                variables.append(feature_events)
            elif config['outliers'] == 'replace_with_median':
                feature_events = feature_events.join(medians, on='variable', how='left').withColumn(
                    'value',
                    F.when(
                        F.col('value').between(config['min'], config['max']),
                        F.col('value')
                    ).otherwise(F.col('median'))
                ).drop('median')
                variables.append(feature_events)
            else:
                raise ValueError(f'Unknown outliers value {config["outliers"]}')
        
        # unite all variables
        events = reduce(DataFrame.unionByName, variables)
        return events
    
    @cache_result('sanitized_events', partitionBy='variable')
    def sanitize_events(self, events: DataFrame, features) -> DataFrame:
        return self.process_outliers(events, features)
    
    def add_demographic_events(self, events, icu_stays):
        age_events = self.extract_age_events(icu_stays)
        patients = self.data_extractor.read_patients()
        gender_events = self.extract_gender_events(icu_stays, patients)
        
        return events.drop('code') \
            .unionByName(age_events) \
            .unionByName(gender_events)
    
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
    
    def use_relative_icu_time(self, events: DataFrame, icu_stays: DataFrame):
        with_rel_time = (
            events.alias('event')
            .join(icu_stays, on='stay_id', how='inner')
            .withColumn('icu_time_minutes',
                        ((F.col('time') - F.col('time_start')) / 60).cast('int'))
            .drop('time')
            .select('event.*', 'icu_time_minutes'))
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
            .filter(F.col('icu_time_minutes') < 24 * 60)
            .select('stay_id')
            .distinct())
        icu_24_plus = icu_24_plus.join(stay_ids_with_events, on='stay_id', how='inner')
        return icu_24_plus
    
    def extract_age_events(self, icu_stays):
        age_events = icu_stays.select(
            'stay_id',
            'admission_id',
            F.col('age').alias('value'),
            F.lit('Age').alias('variable'),
            F.lit(0).alias('icu_time_minutes'),
            F.lit('years').alias('unit'),
            F.lit('Demographics').alias('source'),
        )
        return age_events
    
    def extract_gender_events(self, icu_stays: DataFrame, patients: DataFrame):
        gender_events = icu_stays.join(patients, on='patient_id', how='inner').select(
            'stay_id',
            'admission_id',
            F.when(F.col('GENDER') == 'M', 0).otherwise(1).alias('value'),
            F.lit('Gender').alias('variable'),
            F.lit(0).alias('icu_time_minutes'),
            F.lit(None).alias('unit'),
            F.lit('Demographics').alias('source'),
        )
        return gender_events
    
    def run(self) -> DataFrame:
        features = get_features()
        raw_events = self.get_raw_events(features)
        feature_events = self.process_event_values(raw_events, features)
        sanitized_events = self.sanitize_events(feature_events, features)
        
        # filter ICU stays so we don't generate useless gender and age events
        icu_stays = (self.data_extractor.read_icustays()
                     .join(sanitized_events.select('stay_id').distinct(),
                           on='stay_id',
                           how='inner')).cache()
        admissions = self.data_extractor.read_admissions()
        icu_events = self.use_relative_icu_time(sanitized_events, icu_stays)
        icu_stays_24h = self.get_icu_alive_for_24h(icu_stays, icu_events, admissions)
        
        self.save_mortality_labels(icu_stays_24h)
        self.save_icu_splits(icu_stays_24h)
        
        # for all icu_stays, not only 24h+
        all_events = self.add_demographic_events(icu_events, icu_stays)
        self.save_events(all_events)
        return all_events
    
    def save_icu_splits(self, icu_stays_24h: DataFrame):
        patient_ids = icu_stays_24h.select('patient_id').distinct().cache()
        train_df, test_df, val_df = patient_ids.randomSplit([.64, .2, .16], seed=42)
        for df, name in zip([train_df, test_df, val_df], ['train', 'test', 'val']):
            (icu_stays_24h.join(df, on='patient_id', how='inner').select('stay_id')
             .toPandas().to_numpy().dump(self.outputs[f'{name}_stay_ids']))
        patient_ids.unpersist()
    
    def save_mortality_labels(self, icu_stays_24h: DataFrame):
        # TODO: WHY not to store everything as parquets?
        mortality_labels = (icu_stays_24h.select('stay_id', 'died').toPandas())
        mortality_labels.to_pickle(self.outputs['mortality_labels'])
    
    def save_events(self, events: DataFrame):
        event_triplets = (events
        .select(
            'stay_id',
            'value',
            'variable',
            'source',
            F.col('icu_time_minutes').alias('minute'),
        ))
        
        event_triplets.write.mode('overwrite').parquet(self.outputs['events'])

if __name__ == '__main__':
    spark = get_spark("Preprocessing MIMIC-III dataset")
    data_extractor = DataExtractor(spark)
    job = DataProcessingJob(data_extractor)
    job.run()
