# Setup
```bash
git clone ...
cd ...
echo "layout anaconda $(basename $(pwd))" > .envrc
direnv allow
# wait
pipenv install --dev
```


# Useful Commands

- `snakemake --use-conda --cores 10`: Run the workflow using Conda.
- `--list-changes code`: Show the changes in snakemake jobs.
- `--touch`: Update the output files' timestamps.
- `-n`: Perform a dry-run.
- `-p`: Print the shell command that will be executed.
- `--use-conda`: Use Conda to manage dependencies.
