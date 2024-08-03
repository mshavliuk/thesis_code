from pyspark.sql import (
    SparkSession,
    functions as F,
)

from workflow.scripts.data_processing_job import DataProcessingJob


class TestPlotDistributions:
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
