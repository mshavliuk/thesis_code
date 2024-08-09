from typing import Generator

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data._utils.pin_memory import pin_memory


class Batch(dict[str, torch.Tensor]):
    def __init__(self, *args, **kwargs):
        super(Batch, self).__init__(*args, **kwargs)

    @property
    def model_inputs(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.items() if k in {
            'values', 'times', 'variables', 'input_mask', 'demographics'
        }}

    # def get_expected_output(self):
    #     # Assuming 'labels' is the key for expected output tensors
    #     return self['labels']


class DataLoader(TorchDataLoader):
    """
    DataLoader that moves tensors to device on iteration.
    """
    
    def __init__(self, *args, device, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(device)

    def __iter__(self) -> Generator[Batch, None, None]:
        iterator = super().__iter__()
        for batch in iterator:
            yield Batch({k: v.to(self.device) for k, v in batch.items()})
