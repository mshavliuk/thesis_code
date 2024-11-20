import abc
import re
import warnings
from pathlib import Path
from typing import (
    Literal,
)

import dpath
import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
)
from yaml import (
    SafeLoader,
)

from src.lightning_module import (
    FinetuneModuleConfig,
    PretrainingModuleConfig,
)
from src.util.data_module import (
    DataModuleConfig,
    FinetuneDataModuleConfig,
)


class BaseConfig(BaseModel, abc.ABC, extra='forbid'):
    description: str
    name: str
    wandb_logger: dict | None = Field(
        description="Wandb logger configuration, see lightning.pytorch.loggers.WandbLogger.__init__",
        default=None
    )
    trainer: dict = Field(
        description="Trainer configuration, see lightning.Trainer.__init__"
    )
    checkpoint_callback: dict = Field(
        description="Checkpoint callback configuration, "
                    "see lightning.pytorch.callbacks.ModelCheckpoint.__init__"
    )
    early_stop_callback: dict = Field(
        description="Early stopping callback configuration, "
                    "see lightning.pytorch.callbacks.EarlyStopping.__init__"
    )


class PretrainConfig(BaseConfig, extra='forbid'):
    stage: Literal['pretrain']
    module_config: PretrainingModuleConfig
    data_module_checkpoint: str = Field(
        description="Name of the data module checkpoint to load or create if it does not exist")
    
    data_config: DataModuleConfig


class FinetuneConfig(BaseConfig, extra='forbid'):
    stage: Literal['finetune']
    module_config: FinetuneModuleConfig
    data_module_checkpoint: str = Field(
        description="Name of the data module checkpoint to load")
    module_checkpoint: str = Field(  # TODO: use name + '_pretrain' as default
        description="Name of the model checkpoint (usually from pretrain stage) to load")
    fold_index: int | None = Field(
        description="Fold index to use for cross-validation, if None, the whole dataset is used",
        default=None
    )
    data_fraction: float | None = Field(
        description="Fraction of the data to use, if None, the whole dataset is used",
        default=None
    )
    data_config: FinetuneDataModuleConfig
    
    @model_validator(mode='before')
    def default_module_checkpoint(pre_self: dict) -> dict:
        if not pre_self.get('module_checkpoint') and (name := pre_self.get('name')) is not None:
            pre_self['module_checkpoint'] = f'{name}_pretrain:best'
        return pre_self


type MainConfig = PretrainConfig | FinetuneConfig

env_pattern = re.compile(r'\${(\w+)}')


# TODO: this has to be tested!
def parse_yaml_config_with_includes(file_path: Path, _top_level=True) -> dict:
    with file_path.open('r') as f:
        content_dict = yaml.safe_load(f)
        
        # get yaml file tags
        f.seek(0)
        loader = SafeLoader(f)
        loader.parse_stream_start()
        tags = loader.parse_implicit_document_start().tags or {}
        loader.dispose()
    
    if unknown_tags := set(tags.keys()) - {
        '!include!', '!depends-on!', '!dataset!', '!data-fractions!'
    }:
        warnings.warn(f"Unknown tags {unknown_tags} in {file_path}")
    
    if (include_path := tags.get('!include!')) is not None:
        include_dict = parse_yaml_config_with_includes(
            Path(file_path.parent, include_path).resolve(), _top_level=False)
        
        # avoid reusing name and description from included file
        include_dict.pop('name', None)
        include_dict.pop('description', None)
        
        content_dict = dpath.merge(include_dict, content_dict, flags=dpath.MergeType.REPLACE)
    
    if _top_level and ((depends_on_path := tags.get('!depends-on!')) is not None):
        depends_on_dict = parse_yaml_config_with_includes(
            Path(file_path.parent, depends_on_path).resolve(), _top_level=False)
        
        #   max_events: 880
        #   max_minute: 1440 # 24 * 60
        #   scaler_class: 'VariableStandardScaler'
        #   select_top: 128
        # dataset_config = dpath.search(depends_on_dict, 'data_config/*/dataset/path')
        dataset_config = dpath.search(depends_on_dict, 'data_module_checkpoint')
        
        content_dict = dpath.merge(dataset_config, content_dict)
    
    return content_dict


def read_config(file_path: str) -> MainConfig:
    file = Path(file_path)
    config_dict = {'name': file.stem}  # default name is the file name without extension
    config_dict |= parse_yaml_config_with_includes(file)
    
    try:
        return parse_config(config_dict)
    except (ValidationError, ValueError) as e:
        print(f'Error: Invalid config file {file_path}: {e}')
        exit(1)


def parse_config(config_dict: dict) -> MainConfig:
    try:
        dpath.delete(config_dict, '**/__*__')
    except dpath.exceptions.PathNotFound:
        pass
    
    if config_dict['stage'] == 'pretrain':
        config = PretrainConfig(**config_dict)
    elif config_dict['stage'] == 'finetune':
        config = FinetuneConfig(**config_dict)
    else:
        raise ValueError(f"Invalid stage {config_dict['stage']}")
    return config
