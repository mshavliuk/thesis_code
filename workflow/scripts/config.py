import argparse
import logging
import os
import sys
import tempfile
from dataclasses import (
    dataclass,
    field,
)
from itertools import chain

from pyspark import SparkConf

UNSET = object()


@dataclass
class _cfg:
    _instance = None
    
    output_dir: str = UNSET
    data_dir: str = UNSET
    remote_run: bool = False
    log_level: str = "WARN"
    temp_dir: str = tempfile.gettempdir()
    DEBUG: bool = field(default=False, metadata={'help': 'Enable debug mode'})
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __post_init__(self):
        # if any of the environment variables are set, override the defaults
        env_vars = {k: os.environ.get(k.upper()) for k in self.__dataclass_fields__ if
                    k.upper() in os.environ}
        self.__dict__.update(env_vars)
        parser = argparse.ArgumentParser()
        
        parser.add_argument(
            "--log-level",
            choices=[level for level in logging.getLevelNamesMapping().keys()],
        )
        
        manual_arguments = {
            option.lstrip('-').replace('-', '_')
            for option in chain.from_iterable((action.option_strings for action in parser._actions))
        }
        
        # automatically add all fields as supported cli arguments
        for field_name, data_field in self.__dataclass_fields__.items():
            if field_name not in manual_arguments:
                help_segments = [data_field.metadata.get('help', None)]
                if field_name in env_vars:
                    help_segments.append(f"env value: {field_name.upper()}")
                parser.add_argument(
                    f"--{field_name.replace('_', '-').lower()}",
                    help='\n'.join(filter(None, help_segments)),
                )
        args, _ = parser.parse_known_args()
        
        # update the fields with the parsed arguments
        self.__dict__.update({k: v for k, v in vars(args).items() if v is not None})
        
        # remove unset fields, so the error will be raised if they are not set
        for key in list(self.__dict__.keys()):
            if self.__dict__[key] is UNSET:
                del self.__dict__[key]
                del self.__dataclass_fields__[key]
                del self.__annotations__[key]
        
        for path in ['output_dir', 'data_dir', 'temp_dir']:
            # if not UNSET
            if getattr(self, path) != UNSET:
                setattr(self, path, os.path.abspath(getattr(self, path)))
        
        if os.environ.get('JETBRAINS_REMOTE_RUN') or '.pycharm_helpers' in sys.argv[0]:
            self.DEBUG = True
            os.environ['DEBUG'] = 'True'
            self.remote_run = True
        else:
            pass
    
    def get_spark_conf(self, config: dict = None):
        # TODO: this is not used yet
        if config is None:
            config = {}
        spark_conf = SparkConf()
        spark_conf.setAll(list(config.items()))
        
        if self.DEBUG:
            spark_conf.set("spark.executorEnv.DEBUG", "True")
            
            # if self.remote_run:
            #     spark_conf.set("spark.executorEnv.DEBUG_HOST", os.environ['DEBUG_HOST'])
            #     spark_conf.set("spark.executorEnv.DEBUG_PORT",
            #                    str(int(os.environ['DEBUG_PORT']) + 1))
            #     # spark_conf.set("spark.python.daemon.module", "remote_debug_worker")
        
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


Config = _cfg()
