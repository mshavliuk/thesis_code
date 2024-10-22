import argparse

import wandb
from tqdm import tqdm


def rename_runs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', type=str, required=True, help='Name of the run to rename')
    parser.add_argument('--to', type=str, required=True, help='New name of the run')
    args = parser.parse_args()
    
    api = wandb.Api()
    runs = api.runs("Strats", filters={
        '$or': [
            {"config.name": getattr(args, 'from')},
            {"display_name": getattr(args, 'from')},
        ]
    })
    for run in tqdm(runs):
        run: wandb.apis.public.Run
        run.name = args.to
        run.update()


if __name__ == '__main__':
    rename_runs()
