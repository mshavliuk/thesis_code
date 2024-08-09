import time

import torch
from torch.utils.data import DataLoader

from src.util.collator import Collator
from src.util.dataset import Dataset


class TestDataLoader:
    def test_iterating_over_small_dataset(self, dataset: Dataset):
        data_loader = DataLoader(dataset,
                                 # pinned_memory? persistent_worker?
                                 # pin_memory_device='cuda:0',
                                 # pin_memory=True,
                                 batch_size=1,
                                 shuffle=True,
                                 collate_fn=Collator())
        for _ in data_loader:
            ...
    
    def test_iterating_over_large_dataset(self, large_dataset: Dataset):
        num_workers = 10
        data_loader = DataLoader(large_dataset,
                                 timeout=100000,
                                 batch_size=64,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 collate_fn=Collator())
        ### warmup
        for _ in zip(range(num_workers), data_loader):
            ...
        
        start_time = time.time()
        for _ in data_loader:
            ...
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Time per batch: {total_time / len(data_loader):.4f} seconds {len(data_loader) / total_time}(it/sec)")

