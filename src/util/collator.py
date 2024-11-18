import numpy as np
import torch


class Collator:
    pad_step = 8
    
    def __init__(self, padding_value):
        self.padding_value = padding_value
        self.dtypes = None
        self.collated_keys = ('values', 'variables', 'times')
        self.concat_keys = None
    
    def __call__(self, batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        if self.dtypes is None:
            self.dtypes = {key: torch.from_numpy(val).dtype for key, val in batch[0].items()}
            
            self.concat_keys = tuple(
                key for key in batch[0].keys() if key not in self.collated_keys)
        
        lengths = tuple(len(x['values']) for x in batch)
        max_length = np.ceil(max(lengths) / self.pad_step).astype(np.int16) * self.pad_step
        
        collated = {
            key: torch.full(
                (len(batch), max_length),
                self.padding_value,
                dtype=self.dtypes[key],
                pin_memory=True,
            )
            for key in self.collated_keys
        }
        
        for key in self.collated_keys:
            dest = collated[key].numpy()
            for i, x in enumerate(batch):
                dest[i, :lengths[i]] = x[key]
        
        concatenated = {
            key: torch.from_numpy(
                np.vstack(
                    tuple(x[key] for x in batch),
                    casting='no'
                )
            ).pin_memory()
            for key in self.concat_keys
        }
        
        input_mask = np.arange(max_length).reshape(1, -1) < np.array(lengths).reshape(-1, 1)
        input_mask = torch.from_numpy(input_mask).pin_memory()
        
        return {**collated, **concatenated, 'input_mask': input_mask}
