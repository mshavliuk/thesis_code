from pyspark.sql import (
    DataFrame,
    functions as F,
)
from pyspark.sql.types import StringType

from workflow.scripts.config import Config
from workflow.scripts.constants import get_features
from workflow.scripts.data_processing_job import DataProcessingJob
from workflow.scripts.spark import get_spark


class TestDataProcessingJob(unittest.TestCase):
    def setUp(self):
        self.spark = get_spark()
        self.spark.sparkContext.setLogLevel('WARN')  # FIXME: do this less ugly
        self.data_reader = DataProcessingJob(self.spark)
        self.features = get_features()
        self.show_dataframes = True  # TODO: make configurable
    
    def get_feature(self, name):
        return next(f for f in self.features if f['name'] == name)
    
    def show(self, df):
        if self.show_dataframes:
            df.show(1000)

    
    def test_get_hourly_inputevents_mv(self):
        # Store the original method in a temporary variable
        original_get_inputevents_mv = self.data_reader.read_inputevents_mv
        
        # Override the method using the original method for the actual call to avoid recursion
        # FIXME: MOCK HAS TO BE UNDONE AFTER TEST IS COMPLETE
        self.data_reader.read_inputevents_mv = lambda: original_get_inputevents_mv().limit(100)
        
        hourly_inputevents_mv = self.data_reader.get_hourly_inputevents_mv()
        hourly_inputevents_mv.printSchema()
        self.show(hourly_inputevents_mv)
    

    def test_vasopressin_feature(self):
        feature = next(f for f in self.features if f['name'] == 'Vasopressin')
        inputevents = self.data_reader.get_all_inputevents()
        events = (
            inputevents
            .filter(F.col('code') == feature['codes'][0])
            .withColumn('new_value', feature['select']))
        self.show(events.select('value', 'unit', 'new_value').distinct())
        
        median_value = events.approxQuantile('value', [0.5], 0.1).pop()
        
        iszero = (F.col('value') == 0)
        isunits = F.col('unit').isin(['U', 'units'])
        isinrange = (F.col('value') >= 0) & (F.col('value') <= 400)
        ind = ((isunits & isinrange) | iszero)
        
        outliers = (events.filter(~ind))
        # assert outliers value is set to median
        new_values = outliers.select('new_value').rdd.flatMap(lambda x: x).collect()
        for new_value in new_values:
            self.assertAlmostEqual(new_value, median_value, delta=0.1)
    
    def test_vacomycin_feature(self):
        feature = next(f for f in self.features if f['name'] == 'Vacomycin')
        inputevents = self.data_reader.get_all_inputevents()
        events = (
            inputevents
            .filter(F.col('code') == feature['codes'][0])
            .withColumn('new_value', feature['select']))
        self.show(events.select('value', 'unit', 'new_value').distinct())
        
        median_value = events.approxQuantile('value', [0.5], 0.1).pop()
        
        outliers = events.filter(~(F.col('value').between(0, 8)))
        # assert outliers value is set
        new_values = outliers.select('new_value').rdd.flatMap(lambda x: x).collect()
        for new_value in new_values:
            self.assertAlmostEqual(new_value, median_value, delta=0.1)
        
        self.show(outliers.groupBy('value', 'unit', 'new_value').count())
    
    def test_preprocess(self):
        self.data_reader.preprocess()
    
    # def test_compute_group_statistics(self):
    #     events = self.data_reader.preprocess.get_cached_df(self.spark)
    #     group_stat = self.data_reader.compute_group_statistics(events)
    #     self.show(group_stat)
    #     self.assertIsInstance(group_stat, DataFrame)
    #
    # def test_plot_patient_journeys(self):
    #     events = self.data_reader.preprocess.get_cached_df(self.spark)
    #     group_stat = self.spark.read.csv(
    #         f'{Config.data_dir}/statistics/groups.csv', header=True, )
    #     self.data_reader.plot_patient_journeys(events, group_stat)
