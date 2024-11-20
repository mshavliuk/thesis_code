import tempfile
from pathlib import Path

import pytest
from pyspark.sql import (SparkSession, )

from workflow.scripts.config import Config
from workflow.scripts.data_extractor import DataExtractor
from workflow.scripts.data_plotting_job import DataPlottingJob
from workflow.scripts.latex_data_job import LatexDataJob
from workflow.scripts.statistics_job import StatisticsJob


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
def raw_data_path():
    path = Config.data_dir / 'raw'
    if not path.exists():
        pytest.skip(f"Data directory {path} does not exist")
    yield Path('data')


@pytest.fixture(scope='module')
def data_extractor(spark: SparkSession, raw_data_path: Path):
    yield DataExtractor(spark)


@pytest.fixture(scope='module')
def statistics_job(spark: SparkSession):
    yield StatisticsJob(spark)


@pytest.fixture(scope="module")
def latex_data_job():
    yield LatexDataJob()


@pytest.fixture(scope='module')
def plotting_job(spark: SparkSession):
    yield DataPlottingJob(spark)
