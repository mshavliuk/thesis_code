import time

from src.util.collator import Collator
from src.util.data_loader import DataLoader
from src.util.dataset import (
    PretrainDataset,
    DatasetConfig,
)

config = DatasetConfig(path='/home/user/.cache/thesis/data/preprocessed/train', variables_dropout=0.2)
dataset = PretrainDataset(logger=None, config=config)
exit()
num_workers = 0
data_loader = DataLoader(dataset,
                         # pinned_memory? persistent_worker?
                         timeout=0 if num_workers == 0 else 100000,
                         batch_size=64,
                         shuffle=True,
                         device='cuda:0',
                         num_workers=num_workers,
                         collate_fn=Collator())
### warmup
for _ in zip(range(max(1, num_workers)), data_loader):
    ...

# profiler = cProfile.Profile()
# profiler.enable()
start_time = time.time()
for _ in data_loader:
    ...
# profiler.disable()
end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time:.2f} seconds")
print(f"Time per batch: {total_time / len(data_loader):.4f} seconds {len(data_loader) / total_time}(it/sec)")
# Print the profiling results
# s = StringIO()
# ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
# ps.print_stats()
# print(s.getvalue())
