from itertools import chain

import wandb
from tqdm import tqdm


def sanitise_runs():
    api = wandb.Api()
    
    failed_runs: list[wandb.apis.public.Run] = api.runs("Strats", filters={
        "state": {"$in": ["crashed", "killed"]}
    })
    unfinished_runs: list[wandb.apis.public.Run] = api.runs("Strats", filters={
        "$or": [
            # {"$and": [
            #     {"config.stage": "pretrain"},
            #     {"summary_metrics.test_epoch_loss": {"$exists": False}}, # FIXME: find better way!!!
            # ]},
            {"$and": [
                {"config.stage": "finetune"},
                {"summary_metrics.test_auroc": {"$exists": False}},
            ]},
        ],
        "state": {"$ne": "running"}
    })
    
    for run in tqdm(chain(failed_runs, unfinished_runs),
                    total=len(failed_runs) + len(unfinished_runs)):
        # print(run.name, run.config['stage'], run.summary._json_dict.get('test_epoch_loss'), run.summary._json_dict.get('test_auroc'), run.created_at)
        run.delete()


if __name__ == '__main__':
    sanitise_runs()
