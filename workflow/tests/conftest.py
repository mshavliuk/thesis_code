import tempfile

import pytest
from pyspark.sql import (SparkSession, )

from workflow.scripts.data_extractor import DataExtractor
from workflow.scripts.data_processing_job import DataProcessingJob

@pytest.fixture(scope="session")
def spark_checkpoint_dir():
    with tempfile.TemporaryDirectory() as dir_name:
        yield dir_name

@pytest.fixture(scope="session")
def spark(spark_checkpoint_dir: str):
    # .config("spark.sql.shuffle.partitions", "200") \
    spark = SparkSession.builder.appName("Testing PySpark Example") \
        .master("local[*]") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.sql.parquet.filterPushdown", "true") \
        .config("spark.memory.offHeap.size", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel('INFO')
    spark.sparkContext.setCheckpointDir(spark_checkpoint_dir)
    yield spark


@pytest.fixture(scope='module')
def data_extractor(spark: SparkSession):
    yield DataExtractor(spark)


@pytest.fixture(scope='module')
def data_processing_job(data_extractor: DataExtractor):
    yield DataProcessingJob(data_extractor)
