import logging
import os
import sys
from typing import Callable

from pyspark.sql import DataFrame

from workflow.scripts.config import Config


class CustomLogRecord(logging.LogRecord):
    def getMessage(self):
        msg = str(self.msg)
        str_args = []
        if self.args:
            for arg in self.args:
                if isinstance(arg, DataFrame):
                    str_arg = "(total %s)\n%s"
                    was_cached = arg.is_cached
                    if not was_cached:
                        arg.cache()
                    
                    str_args.append(str_arg % (arg.count(), arg._jdf.showString(1000, 80, False)))
                    
                    if not was_cached:
                        arg.unpersist()
                elif isinstance(arg, Callable):
                    str_args.append(arg())
                else:
                    str_args.append(str(arg))
            msg = msg % tuple(str_args)
        return msg


def _setup_logger(level) -> logging.Logger:
    logging.basicConfig(level=logging.WARN)
    logging.captureWarnings(True)
    logging.setLogRecordFactory(CustomLogRecord)
    logger = logging.getLogger('app_logger')
    logger.setLevel(level)
    logger.propagate = False
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    
    class StdoutFilter(logging.Filter):
        def filter(self, record):
            return record.name == 'app_logger' and record.levelno <= logging.INFO
    
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(StdoutFilter())
    logger.addHandler(stdout_handler)
    
    class StderrFilter(logging.Filter):
        def filter(self, record):
            return record.name == 'app_logger' and record.levelno > logging.INFO
    
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.addFilter(StderrFilter())
    logger.addHandler(stderr_handler)
    
    logger.warning("WARN logging enabled")
    logger.debug("DEBUG logging enabled")
    
    return logger


def get_logger() -> logging.Logger:
    logger = logging.getLogger('app_logger')
    if not logger.handlers:
        return _setup_logger(level=Config.log_level)
    return logger
