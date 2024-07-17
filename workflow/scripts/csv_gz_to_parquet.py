import argparse
import logging
import pathlib

from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("input_file_path", help="path to the input file")
parser.add_argument("output_file_path", help="path to the output file")
args = parser.parse_args()

input_file = pathlib.Path(args.input_file_path)
output_file = pathlib.Path(args.output_file_path)

spark = SparkSession.builder \
    .appName(f"Converting {input_file} to {output_file}") \
    .config("spark.executor.cores", "1") \
    .config("spark.executor.instances", "8") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED") \
    .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()

df = spark.read.csv(str(input_file.absolute()), header=True, inferSchema=True)
df.write.parquet(str(output_file.absolute()))

spark.stop()
