import warnings

import wandb


def get_run_checkpoint(run: wandb.apis.public.Run, return_artifact=False):
    model_artifacts = [artifact for artifact in run.logged_artifacts() if
                       artifact.type in {'checkpoint', 'model'}]
    return get_checkpoint_from_artifacts(model_artifacts, return_artifact)
    

def get_checkpoint_from_artifacts(artifacts: list[wandb.Artifact], return_artifact=False):
    if len(artifacts) > 1:
        warnings.warn("Multiple model artifacts found. Using the last one", stacklevel=2)
    elif len(artifacts) == 0:
        raise FileNotFoundError("Artifact does not contain checkpoint")
    
    artifact = artifacts[-1]
    
    assert len(artifact.manifest.entries) > 0, "Artifact does not contain any entries"
    assert artifact.type in {'checkpoint', 'model'}, "Artifact is not a checkpoint or model"
    
    entry_name = next(iter(artifact.manifest.entries))
    
    checkpoint = artifact.get_entry(entry_name).download()
    if return_artifact:
        return checkpoint, artifact
    else:
        return checkpoint
