from datetime import datetime

import pytest
from pyspark.sql import (
    functions as F,
)

from workflow.scripts.data_processing_job import DataProcessingJob
from workflow.tests.conftest import data_processing_job


@pytest.fixture(scope="function", name='obj')
def data_processing_job_shortcut(data_processing_job: DataProcessingJob):
    return data_processing_job


class TestSplitAmountHourly:
    @pytest.fixture(scope="function")
    def split_amount_hourly(self, obj: DataProcessingJob):
        return obj.distribute_value_in_time_udf.func
    
    def test_with_exact_hours(self, split_amount_hourly):
        result = split_amount_hourly(100, datetime(2020, 1, 1, 10), 2)
        expected = [
            (datetime(2020, 1, 1, 9), 50.0),
            (datetime(2020, 1, 1, 10), 50.0)
        ]
        assert list(result) == expected
    
    def test_with_partial_hour(self, split_amount_hourly):
        result = split_amount_hourly(100, datetime(2020, 1, 1, 10), 2.5)
        expected = [
            (datetime(2020, 1, 1, 8, 30), 40.0),
            (datetime(2020, 1, 1, 9, 30), 40.0),
            (datetime(2020, 1, 1, 10), 20.0)]
        assert list(result) == expected
    
    def test_with_less_than_one_hour(self, split_amount_hourly):
        result = split_amount_hourly(50, datetime(2020, 1, 1, 1), 0.5)
        expected = [(datetime(2020, 1, 1, 1), 50.0)]
        
        assert list(result) == expected
    
    def test_with_zero_hours(self, split_amount_hourly):
        result = split_amount_hourly(100, datetime(2020, 1, 1, 10), 0)
        expected = [(datetime(2020, 1, 1, 10), 100)]
        
        assert list(result) == expected
    
    def test_with_negative_hours(self, split_amount_hourly):
        # this looks unexpected, but it's the way it was treated in the original paper
        result = split_amount_hourly(100, datetime(2020, 1, 1, 10), -1)
        expected = [(datetime(2020, 1, 1, 10), 100)]
        
        assert list(result) == expected


class TestProcessOutliers:
    @pytest.fixture(scope="function")
    def features(self):
        return [
            {
                'name': 'Temperature',
                'codes': [1, 2, 3],
                'min': 14.2,
                'max': 47,
                'outliers': 'remove',
            },
            {
                'name': 'HR',
                'codes': [4, 5, 6],
                'min': 40,
                'max': 200,
                'outliers': 'replace_with_median',
            }
        ]
    
    @pytest.fixture(scope="module")
    def df(self, spark):
        dt = datetime(2020, 1, 1, 1)
        return spark.createDataFrame([
            (1, 36.6, dt, 'Temperature'),
            (2, 37.1, dt, 'Temperature'),
            (3, 36.9, dt, 'Temperature'),
            (3, 1000.0, dt, 'Temperature'),
            (4, 1.0, dt, 'HR'),
            (5, 100.0, dt, 'HR'),
            (6, 300.0, dt, 'HR'),
        ], ['code', 'value', 'time', 'group'])
    
    def test_process_outliers(self, obj: DataProcessingJob, df, features):
        result = obj.process_outliers(df, features)
        assert set(result.columns) == {'code', 'value', 'time', 'group'}
        result_values = result.orderBy('code').rdd.map(lambda x: x.value).collect()
        assert result_values == [36.6, 37.1, 36.9, 100, 100, 100]
    
    def test_partial_filtering_config_raises_error(self, obj: DataProcessingJob, df, features):
        for config_key in 'min', 'max', 'outliers':
            config = features[0].copy()
            del config[config_key]
            with pytest.raises(ValueError,
                               match="Filtering config for group Temperature should contain ..."):
                obj.process_outliers(df, [config])
    
    def test_distinct_config_for_same_group_raises_error(
        self,
        obj: DataProcessingJob,
        df,
        features
    ):
        features = [
            features[0],
            features[0].copy(),
            features[1],
        ]
        features[1]['min'] += 1
        with pytest.raises(ValueError,
                           match="Filtering configs for group Temperature are different"):
            obj.process_outliers(df, features)


class TestApplyFeatureSelectors:
    @pytest.fixture(scope="function")
    def features(self):
        return [{
            # Celsius
            'name': 'Temperature',
            'codes': [677],
        },
            {
                # Fahrenheit
                'name': 'Temperature',
                'codes': [678],
                'select': (F.col('value') - 32) * 5 / 9,
            }
        ]
    
    @pytest.fixture(scope="module")
    def df(self, spark):
        dt = datetime(2020, 1, 1, 1)
        return spark.createDataFrame([
            (677, 36.6, dt, 'Temperature'),
            (677, 37.1, dt, 'Temperature'),
            (677, 36.9, dt, 'Temperature'),
            (678, 98.24, dt, 'Temperature'),
            (678, 97.88, dt, 'Temperature'),
            (678, 98.12, dt, 'Temperature'),
        ], ['code', 'value', 'time', 'group'])
    
    def test_with_colliding_group_names(self, obj: DataProcessingJob, df, features):
        result = obj.apply_feature_selectors(df, features)
        assert result.count() == 6
        assert result.columns == ['code', 'value', 'time', 'group', 'feature_value']
        result_values = result.orderBy('code').rdd.map(lambda x: x.feature_value).collect()
        for expected, actual in zip([36.6, 37.1, 36.9, 36.8, 36.6, 36.73], result_values):
            assert pytest.approx(expected, 0.01) == actual

class TestPreprocess:
    def test_preprocess(self, obj: DataProcessingJob):
        obj.preprocess()
