from itertools import (
    chain,
    product,
)
from pathlib import Path
from typing import cast

from snakemake.rules import Rule
from snakemake.workflow import Workflow
from yaml import SafeLoader

def fold_range(*args):
    return range(1,config["number_of_folds"] + 1)

def all_pretrain_configs(*args, **kwargs):
    return glob_wildcards("experiments/pretrain/{file}.yaml").file

def all_finetune_configs(*args, **kwargs):
    return glob_wildcards("experiments/finetune/{file}.yaml").file

def get_config_tags(config_path: Path) -> dict:
    with config_path.open('r') as f:
        loader = SafeLoader(f)
        loader.parse_stream_start()
        tags = loader.parse_implicit_document_start().tags
        loader.dispose()
    return tags or {}


def get_config_dependency(config_path: Path):
    tags = get_config_tags(config_path)
    if '!depends-on!' not in tags:
        return None
    return Path(config_path.parent,tags['!depends-on!']).resolve()


def get_test_pretrain_dependent_paths(wildcards):
    pretrain_output = [
        workflow.get_rule('pretrain').expand_output(
            wildcards={'file': wildcards.file, 'fold': f}
        )[0] for f in fold_range()
    ]
    return chain.from_iterable(pretrain_output)

def get_pretrain_all_dependent_paths(*args, **kwargs):
    pretrain_output = [
        workflow.get_rule('pretrain').expand_output(
            wildcards={'file': file, 'fold': fold}
        )[0] for file, fold in product(
            all_pretrain_configs(), fold_range()
        )
    ]
    return chain.from_iterable(pretrain_output)

def get_finetune_config_dependent_paths(wildcards):
    finetune_output = [
        workflow.get_rule('finetune').expand_output(
            wildcards={'file': wildcards.file, 'fraction': f}
        )[0] for f in get_finetune_data_fractions(wildcards)
    ]

    return chain.from_iterable(finetune_output)

def get_finetune_all_dependent_paths(*args, **kwargs):
    finetune_output = [
        workflow.get_rule('finetune_config').expand_output(
            wildcards={'file': file}
        )[0] for file in all_finetune_configs()
    ]
    return chain.from_iterable(finetune_output)


def get_finetune_depend_on_pretrain_status(wildcards):
    config_file = Path(f"experiments/finetune/{wildcards.file}.yaml")
    dependency = get_config_dependency(config_file)
    return checkpoints.test_pretrain.get(file=dependency.stem).output


def dict_to_cli_args(d: dict):
    return " ".join(
        f"--{key}" if isinstance(value,bool) and value else f"--{key}={value}"
        for key, value in d.items() if value is not False
    )

    return " ".join(f"--{key}={value}" for key, value in d.items())


def get_finetune_data_fractions(wildcards):
    config_file = Path(f"experiments/finetune/{wildcards.file}.yaml")
    tags = get_config_tags(config_file)
    if (data_fractions := tags.get('!data-fractions!')) is not None:
        return [float(f) for f in data_fractions.split(",")]
    return config["data_fractions"]


def get_all_datasets_and_statuses(_):
    """
    This is used as rule input to ensure all datasets are generated successfully
    and allows to avoid using partially generated datasets
    """
    generate_dataset_outputs = [
        checkpoints.generate_dataset.get(dataset_name=d).output for d in
        config["datasets"]
    ]

    return chain.from_iterable(generate_dataset_outputs)


def wildcard_validator(wildcards):
    if (fraction := getattr(wildcards,"fraction",None)) is not None:
        assert 0 < float(fraction) <= 1, f"Invalid fraction {fraction}: must be in the range (0, 1]"


def apply_defaults():
    for _rule in cast(Workflow,workflow).rules:
        _rule: Rule
        _rule.params.update({'wildcard_validator': wildcard_validator})
