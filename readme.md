# Setup
```bash
git clone ...
cd ...

conda update -n base -c defaults conda # optional

conda env create -f environment.yml
conda activate ...

conda env config vars set PYTHONPATH=$(pwd) DATA_DIR=$(pwd)/data WANDB_PROJECT=Cluster TEMP_DIR=/tmp/srmish/ MPLCONFIGDIR=/tmp/srmish/matplotlib

# see https://snakemake.readthedocs.io/en/latest/executing/cli.html#profiles
conda env config vars set SNAKEMAKE_PROFILE=$(pwd)/workflow/profiles/workstation # in case of local usage
# or
conda env config vars set SNAKEMAKE_PROFILE=$(pwd)/workflow/profiles/tuni-cluster # in case of slurm tuni cluster



wandb login
```



# Useful Commands

- `snakemake --use-conda --cores 10`: Run the workflow using Conda.
- `--list-changes code`: Show the changes in snakemake jobs.
- `--touch`: Update the output files' timestamps.
- `-n`: Perform a dry-run.
- `-p`: Print the shell command that will be executed.
- `--use-conda`: Use Conda to manage dependencies.
- `snakemake --lint`: apply formatting rules
