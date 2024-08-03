from pyspark import SparkConf
from pyspark.sql import SparkSession

from workflow.scripts.config import Config


def get_spark(app_name: str, spark_conf: SparkConf = None) -> SparkSession:
    if spark_conf is None:
        spark_conf = Config.get_spark_conf()
        
    # TODO: allow to customize app name
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config(conf=spark_conf) \
        .getOrCreate()
    # set checkpoint dir
    spark.sparkContext.setCheckpointDir(f"{Config.temp_dir}/checkpoints")
    spark.sparkContext.setLogLevel(Config.log_level)
    return spark
