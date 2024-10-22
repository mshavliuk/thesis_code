from collections import defaultdict

import numpy as np
import torch


class Collator:
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
        max_length = np.ceil(max(lengths) / 8).astype(np.int16) * 8
        
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


class NestedTensorCollator:
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
        
        collated = []
        
        concatenated = {
            key: torch.zeros(len(batch), len(batch[0][key]), dtype=self.dtypes[key])
            for key in self.concat_keys}
        
        for i, x in enumerate(batch):
            collated.append([])
            for key in self.collated_keys:
                collated[i].append(torch.from_numpy(x[key]))
            collated[i] = torch.stack(collated[i]).permute(1, 0)
            for key in self.concat_keys:
                concatenated[key][i] = torch.from_numpy(x[key])
        
        return {
            'triplets': torch.nested.nested_tensor(collated, layout=torch.jagged),
            **concatenated,
        }


class JaggedCollator:
    collated_keys = ('values', 'times', 'variables')
    
    def __init__(self):
        self.dtypes = None
        self.concat_keys = None
    
    def __call__(self, batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        if len(batch) == 0:
            return {}
        if self.dtypes is None:
            import fbgemm_gpu  # noqa
            self.dtypes = {key: torch.from_numpy(val).dtype for key, val in batch[0].items()}
        if self.concat_keys is None:
            self.concat_keys = tuple(
                key for key in batch[0].keys() if key not in self.collated_keys)
        
        collated_lists = defaultdict(list)
        
        # concatenated = {
        #     key: torch.zeros(len(batch), len(batch[0][key]), dtype=self.dtypes[key],
        #                      pin_memory=True,
        #                      requires_grad=False)
        #     for key in self.concat_keys}
        concat_lists = defaultdict(list)
        
        for x in batch:
            for key in self.collated_keys:
                collated_lists[key].append(torch.from_numpy(x[key]))
            for key in self.concat_keys:
                concat_lists[key].append(torch.from_numpy(x[key]))
        
        concatenated = {
            key: torch.zeros(len(batch), len(batch[0][key]), dtype=self.dtypes[key],
                             pin_memory=True,
                             requires_grad=False)
            for key in self.concat_keys}
        
        lengths = [len(x) for x in collated_lists['values']]
        cum_length = sum(lengths)
        collated = {key: torch.zeros(cum_length, dtype=self.dtypes[key],
                                     pin_memory=True,
                                     requires_grad=False)
                    for key in self.collated_keys}
        
        for key, values in concat_lists.items():
            torch.vstack(values, out=concatenated[key])
        
        for key, values in collated_lists.items():
            torch.hstack(values, out=collated[key])
        
        lengths = torch.tensor(lengths, dtype=torch.int16,
                               pin_memory=True,
                               requires_grad=False)
        # max_length = np.ceil(lengths.max().item() / 8) * 8
        return {
            **collated,
            **concatenated,
            'lengths': lengths,
            # 'max_length': max_length,
        }


class JaggedTripletCollator:
    
    def __init__(self):
        self.dtypes = None
        self.keys = None
        self.collated_keys = ('values', 'times', 'variables')
        self.concat_keys = None
    
    def __call__(self, batch: list[dict[str, np.ndarray]], length=None) -> dict[str, torch.Tensor]:
        if len(batch) == 0:
            return {}
        if self.dtypes is None:
            import fbgemm_gpu  # noqa
            self.dtypes = {key: torch.from_numpy(val).dtype for key, val in batch[0].items()}
        if self.keys is None:
            self.keys = batch[0].keys()
        if self.concat_keys is None:
            self.concat_keys = tuple(key for key in self.keys if key not in self.collated_keys)
        
        collated = []
        
        concatenated = {
            key: torch.zeros(len(batch), len(batch[0][key]), dtype=self.dtypes[key])
            for key in self.concat_keys}
        
        # for collated tensors - join them into a single tensor
        # then concat
        for i, x in enumerate(batch):
            collated.append([])
            for key in self.collated_keys:
                collated[i].append(torch.from_numpy(x[key]))
            collated[i] = torch.stack(collated[i]).permute(1, 0)
            for key in self.concat_keys:
                concatenated[key][i] = torch.from_numpy(x[key])
        
        lengths = torch.tensor([len(x) for x in collated])
        collated = torch.vstack(collated)
        max_length = length or np.ceil(max(
            len(x['values']) for x in batch) / 8).astype(np.int16) * 8
        return {
            'triplets': collated,
            'lengths': lengths,
            **concatenated,
            'max_length': max_length,
        }
