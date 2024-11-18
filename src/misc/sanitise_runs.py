import wandb
from tqdm import tqdm


def sanitise_runs():
    api = wandb.Api()
    
    runs = api.runs("Strats", filters={
        "state": {"$in": ["crashed", "failed"]}
    })
    for run in tqdm(runs):
        run: wandb.apis.public.Run
        run.delete(delete_artifacts=True)


if __name__ == '__main__':
    sanitise_runs()
