import os

from pyspark import SparkConf
from pyspark.sql import SparkSession

from workflow.scripts.config import Config


def get_spark_conf():
    spark_conf = SparkConf()
    
    if Config.debug:
        spark_conf.set("spark.executorEnv.DEBUG", "True")
        
        if Config.debugger_attached:
            spark_conf.set("spark.executorEnv.DEBUG_HOST", os.environ['DEBUG_HOST'])
            spark_conf.set("spark.executorEnv.DEBUG_PORT",
                           str(int(os.environ['DEBUG_PORT']) + 1))
            spark_conf.set("spark.python.daemon.module", "remote_debug_worker")
    
    spark_conf.set("spark.ui.enabled", "false")
    spark_conf.set("spark.driver.memory", "10g")
    spark_conf.set('spark.driver.cores', '4')
    spark_conf.set("spark.executor.cores", "2")
    spark_conf.set("spark.executor.instances", "4")
    spark_conf.set("spark.executor.memory", "4g")
    spark_conf.set("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
    spark_conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
    
    spark_conf.set("spark.executor.heartbeatInterval", "60s")
    spark_conf.set("spark.network.timeout", "600s")
    spark_conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    return spark_conf


def get_spark(app_name: str, spark_conf: SparkConf = None) -> SparkSession:
    if spark_conf is None:
        spark_conf = get_spark_conf()
        
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config(conf=spark_conf) \
        .getOrCreate()
    
    # set checkpoint dir
    spark.sparkContext.setCheckpointDir(f"{Config.temp_dir}/checkpoints")
    spark.sparkContext.setLogLevel(Config.spark_log_level)
    return spark
