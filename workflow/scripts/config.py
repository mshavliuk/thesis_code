import os
import sys
import tempfile
from typing import Literal

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
)


class ConfigClass(BaseModel):
    # TODO: allow define config as cli args
    
    data_dir: DirectoryPath = Field(
        default=os.getenv('DATA_DIR', './data'),
        description="Directory containing the project data",
        validate_default=True)
    
    temp_dir: DirectoryPath = Field(
        default=os.getenv('TEMP_DIR', tempfile.gettempdir()),
        description="Directory for temporary files",
        validate_default=True)
    
    log_level: Literal['ERROR', 'WARN', 'INFO', 'DEBUG'] = Field(
        default='INFO',
        description="Logging level",
    )
    
    spark_log_level: Literal['ERROR', 'WARN', 'INFO', 'DEBUG'] = Field(
        default='WARN',
        description="Logging level",
    )
    
    debug: bool = Field(default=False, description="Enable debug mode")
    
    debugger_attached: bool = Field(
        default_factory=lambda: os.getenv('JETBRAINS_REMOTE_RUN', False) or '.pycharm_helpers' in
                                sys.argv[0],
        description="Is the code running in a remote environment"
    )


Config = ConfigClass()
