import functools
from typing import (
    Callable,
    get_type_hints,
)

from pyspark.errors import (
    AnalysisException,
    # PySparkAssertionError,
)
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType
# from pyspark.testing import assertSchemaEqual

from workflow.scripts.config import Config


def cache_result(cache_key: str = None, cache_dir: str =None,**cache_kwargs):
    """
    A decorator to persist the DataFrame returned by a function to disk.

    # TODO: add some simple hashing/signature to invalidate stale cache
    # RESEARCH: Wouldn't .persist(StorageLevel.DISK) have similar effect?
    """
    cache_dir = cache_dir or Config.temp_dir
    
    def decorator_cache_as(func: Callable[[...], DataFrame]):
        if get_type_hints(func).get('return') is not DataFrame:
            raise TypeError(f"The decorated function {func.__name__} must be annotated to return a DataFrame.")
        
        key = cache_key or func.__name__
        cache_path = f"{cache_dir}/{key}.parquet"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # args_keys = [] # TODO: hash or pickle and compare to invalidate cache
            # for arg in args:
            #     if is_hashable(arg):
            #         args_keys.append(hash(arg))
            #     else: # serialize and hash
            #         args_keys.append(hash(str(arg)))
            # if args or kwargs:
                # self.logger.warning(
                #     f"Arguments passed to {func.__name__} will be ignored when loading from cache."
                # )
            
            df = func(*args, **kwargs)
            return cache_df(df, cache_path, **cache_kwargs)
        
        wrapper.cache_key = key
        wrapper.cache_path = cache_path
        
        def get_cached_df(spark):
            return spark.read.parquet(cache_path)
        
        wrapper.get_cached_df = get_cached_df
        
        return wrapper
    
    return decorator_cache_as


def cache_df(df: DataFrame, path: str, **kwargs) -> DataFrame:
    """
    Persist a DataFrame to disk as a parquet file or load it from disk if it already exists.
    """
    spark = df.sparkSession
    try:
        cached_df = spark.read.parquet(path)
        if not compare_schemas(cached_df.schema, df.schema):
            raise ValueError
        return cached_df
    except (AnalysisException, ValueError):
        df.write.parquet(path, compression='snappy', mode='overwrite', **kwargs)
        return spark.read.parquet(path)

def compare_schemas(s1: StructType, s2: StructType) -> bool:
    schema1 = {(f.name, str(f.dataType)) for f in sorted(s1.fields, key=lambda x: x.name)}
    schema2 = {(f.name, str(f.dataType)) for f in sorted(s2.fields, key=lambda x: x.name)}
    return schema1 == schema2
    # try:
    #     assertSchemaEqual(s1, s2)
    #     return True
    # except PySparkAssertionError:
    #     return False

def is_hashable(variable):
    try:
        hash(variable)
    except TypeError:
        return False
    return True
