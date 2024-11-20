import tempfile
from datetime import datetime

import pytest
from pyspark.sql import (
    functions as F,
)

from workflow.scripts.data_extractor import DataExtractor
from workflow.scripts.data_processing_job import DataProcessingJob


@pytest.fixture(scope="function", name='obj')
def obj(data_extractor: DataExtractor):
    with tempfile.TemporaryDirectory() as dir_name:
        yield DataProcessingJob(data_extractor, dir_name)


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
    
    @pytest.mark.parametrize("amount,time,num_hours,expected_result", [
        (100, datetime(2020, 1, 1, 10), 2.5, [
            (datetime(2020, 1, 1, 8, 30), 40.0),
            (datetime(2020, 1, 1, 9, 30), 40.0),
            (datetime(2020, 1, 1, 10), 20.0)
        ]),
        (100, datetime(2021, 1, 1, 10), 4, [
            (datetime(2021, 1, 1, 7), 25.0),
            (datetime(2021, 1, 1, 8), 25.0),
            (datetime(2021, 1, 1, 9), 25.0),
            (datetime(2021, 1, 1, 10), 25.0)
        ]),
    ])
    def test_with_partial_hour(self, split_amount_hourly, amount, time, num_hours, expected_result):
        result = split_amount_hourly(amount, time, num_hours)
        assert list(result) == expected_result
    
    def test_with_less_than_one_hour(self, split_amount_hourly):
        result = split_amount_hourly(50, datetime(2020, 1, 1, 1), 0.5)
        expected = [(datetime(2020, 1, 1, 1), 50.0)]
        
        assert list(result) == expected
    
    def test_less_than_5_min_tail(self, split_amount_hourly):
        result = split_amount_hourly(121.5, datetime(2022, 1, 1, 10, 3), 4.05)
        expected = [
            (datetime(2022, 1, 1, 7), 30.0),
            (datetime(2022, 1, 1, 8), 30.0),
            (datetime(2022, 1, 1, 9), 30.0),
            (datetime(2022, 1, 1, 10, 3), 31.5),
        ]
        
        assert [(time.replace(microsecond=0), pytest.approx(amount))
                for time, amount in result] == expected
    
    def test_with_zero_hours(self, split_amount_hourly):
        result = split_amount_hourly(100, datetime(2020, 1, 1, 10), 0)
        expected = [(datetime(2020, 1, 1, 10), 100)]
        
        assert list(result) == expected
    
    def test_with_negative_hours(self, split_amount_hourly):
        result = split_amount_hourly(100, datetime(2020, 1, 1, 10), -1)
        
        assert list(result) == []


class TestProcessOutliers:
    @pytest.fixture(scope="function")
    def features(self, spark):
        return [
            {
                'name': 'Temperature',
                'codes': [1, 2, 3],
                'filtering': {'valid': F.col('value').between(14, 47), 'action': 'remove'},
            },
            {
                'name': 'HR',
                'codes': [4, 5, 6],
                'filtering': {'valid': F.col('value').between(40, 200),
                              'action': 'replace_with_median'},
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
        ], ['code', 'value', 'time', 'variable'])
    
    def test_process_outliers(self, obj: DataProcessingJob, df, features):
        result = obj.process_outliers(df, features)
        assert set(result.columns) == {'code', 'value', 'time', 'variable'}
        result_values = result.orderBy('code').rdd.map(lambda x: x.value).collect()
        assert result_values == [36.6, 37.1, 36.9, 100, 100, 100]


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
        ], ['code', 'value', 'time', 'variable'])
    
    def test_with_colliding_group_names(self, obj: DataProcessingJob, df, features):
        result = obj.apply_feature_selectors(df, features)
        assert result.count() == 6
        assert result.columns == ['code', 'value', 'time', 'variable', 'feature_value']
        result_values = result.orderBy('code').rdd.map(lambda x: x.feature_value).collect()
        for expected, actual in zip([36.6, 37.1, 36.9, 36.8, 36.6, 36.73], result_values):
            assert pytest.approx(expected, 0.01) == actual
