import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from src.util.config import MainConfig


def get_callbacks(config: MainConfig) -> list[L.Callback]:
    early_stop = EarlyStopping(**config.early_stop_callback)
    
    checkpoint_callback = ModelCheckpoint(
        **config.checkpoint_callback,
        filename='strats-{epoch}-{val_epoch_loss:.2f}',
        verbose=True,
    )
    
    return [early_stop, checkpoint_callback]
