import tempfile
import unittest
from datetime import (
    datetime,
)

import numpy as np
import pandas as pd
from pyspark.sql import (
    DataFrame,
    SparkSession,
    functions as F,
)

from workflow.scripts.data_processing_job import DataProcessingJob
from workflow.scripts.misc import (
    apply_feature_selectors,
    plot_distributions,
    process_outliers,
    split_amount_hourly,
)
from workflow.scripts.plotting_pandas import (
    get_plot_measurements_distribution,
    get_plot_patient_journey,
)



class TestPlotMeasurementsDistribution(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSession.builder.appName("TestSession").getOrCreate()
        self.df = pd.read_csv('../tests/data/sbp_events.csv')
        codes = self.df['code'].unique()
        self.group_data = pd.DataFrame({
            'code': codes,
            'min': [0] * len(codes),
            'max': [150] * len(codes),
            'group': ['SBP'] * len(codes),
            'name': [
                'Manual BP [Systolic]',
                'Manual Blood Pressure Systolic Left',
            ]
        })
    
    def test_plot_and_save(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_and_save, schema = get_plot_measurements_distribution(temp_dir)
            result = plot_and_save(('SBP',), self.df, self.group_data)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape, (1, 3))
            self.assertEqual(result.iloc[0, 0], "SBP")
            self.assertEqual(result.iloc[0, 1], f"{temp_dir}/hist_SBP.png")
    
    def test_plot_and_save_with_nan_min_max(self):
        group_data = self.group_data.copy()
        group_data['min'] = np.nan
        group_data['max'] = np.nan
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_and_save, schema = get_plot_measurements_distribution(temp_dir)
            result = plot_and_save(('SBP',), self.df, group_data)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape, (1, 3))
            self.assertEqual(result.iloc[0, 0], "SBP")
            self.assertEqual(result.iloc[0, 1], f"{temp_dir}/hist_SBP.png")
    
    def test_plot_and_save_without_codes(self):
        df = self.df.copy()
        df['code'] = np.nan
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_and_save, schema = get_plot_measurements_distribution(temp_dir)
            result = plot_and_save(('SBP',), df, self.group_data)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape, (1, 3))
            self.assertEqual(result.iloc[0, 0], "SBP")
            self.assertEqual(result.iloc[0, 1], f"{temp_dir}/hist_SBP.png")


class TestPlotDistributions(unittest.TestCase):
    def setUp(self):
        codes = {442, 3317, 3323, 3321, 224167}
        self.spark = SparkSession.builder.appName("TestSession").getOrCreate()
        feature_events_path = '/home/user/.cache/thesis/data/preprocessed/unfiltered_feature_events.parquet'
        self.df = self.spark.read.parquet(feature_events_path).filter(F.col('code').isin(codes))
        self.features = [
            {
                'name': 'SBP',
                'codes': codes,
                'min': 0,
                'max': 375,
            }
        ]
        data_reader = DataProcessingJob(self.spark)
        self.items = data_reader.read_items()
    
    def test_plot_distributions(self):
        plot_distributions(self.df, self.items, self.features).collect()
    
    def test_plot_distributions_with_null_codes(self):
        df = self.df.withColumn('code', F.lit(None))
        plot_distributions(df, self.items, self.features).collect()


class TestPlotPatientJourney(unittest.TestCase):
    def setUp(self):
        self.events = pd.read_csv('../tests/data/patient_events.csv',
                                  parse_dates=['time'])
        self.statistics = pd.read_csv('../tests/data/groups_statistics.csv')
    
    def test_plot_patient_journey_tall(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_func, schema = get_plot_patient_journey(temp_dir, self.statistics)
            plot_func((107032,), self.events)
    
    def test_plot_patient_journey_short(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_func, schema = get_plot_patient_journey(temp_dir, self.statistics)
            codes = self.events['code'].unique()[:10]
            events = self.events[self.events['code'].isin(codes)]
            plot_func((107032,), events)
