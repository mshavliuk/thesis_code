# WORK IN PROGRESS


# Setup
```bash
git clone ...
cd ...

conda env create -f environment.yml
conda activate ...

conda env config vars set PYTHONPATH=$(pwd) DATA_DIR=$(pwd)/data WANDB_PROJECT=Cluster TEMP_DIR=/tmp/srmish/ MPLCONFIGDIR=/tmp/srmish/matplotlib

# see https://snakemake.readthedocs.io/en/latest/executing/cli.html#profiles
conda env config vars set SNAKEMAKE_PROFILE=$(pwd)/workflow/profiles/workstation # in case of local usage
# or
conda env config vars set SNAKEMAKE_PROFILE=$(pwd)/workflow/profiles/tuni-cluster # in case of slurm tuni cluster


# required for wandb checkpoints and logging
wandb login
```


