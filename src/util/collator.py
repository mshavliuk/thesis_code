import numpy as np
import torch


class Collator:
    def __init__(self):
        self.dtypes = None
        self.keys = None
        self.collated_keys = ('values', 'variables', 'times')
        self.concat_keys = None
    
    def __call__(self, batch: list[dict[str, np.ndarray]], length=None) -> dict[str, torch.Tensor]:
        if len(batch) == 0:
            return {}
        if self.dtypes is None:
            self.dtypes = {key: torch.from_numpy(val).dtype for key, val in batch[0].items()}
        if self.keys is None:
            self.keys = batch[0].keys()
        if self.concat_keys is None:
            self.concat_keys = tuple(key for key in self.keys if key not in self.collated_keys)
        
        max_length = length or max(len(x['values']) for x in batch)
        collated = {
            key: torch.zeros(len(batch), max_length, dtype=self.dtypes[key])
            for key in self.collated_keys
        }
        
        concatenated = {
            key: torch.zeros(len(batch), len(batch[0][key]), dtype=self.dtypes[key])
            for key in self.concat_keys}
        
        input_mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
        
        for i, x in enumerate(batch):
            input_mask[i, :len(x['values'])] = True
            for key in self.collated_keys:
                collated[key][i, :len(x[key])] = torch.from_numpy(x[key])
            for key in self.concat_keys:
                concatenated[key][i] = torch.from_numpy(x[key])
        
        # TODO: think about https://pytorch.org/docs/stable/nested.html
        
        return {**collated, **concatenated, 'input_mask': input_mask}
        # return {
        #     'input_mask': input_mask,
        #     'values': collated['values'],
        #     'times': collated['times'],
        #     'variables': collated['variables'],
        #     'demographics': concatenated['demographics'],
        # }, {
        #     'forecast_values': concatenated['forecast_values'],
        #     'forecast_mask': concatenated['forecast_mask']
        # }
