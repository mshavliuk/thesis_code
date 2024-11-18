import argparse
import time
from pathlib import Path

from tqdm import tqdm

from src.util.config import read_config
from src.util.data_module import MIMICIIIDataModule

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()
config = read_config(args.config)

checkpoint_dir = Path('/home/user/.cache/thesis/checkpoints/finetune')
# latest dir
checkpoint_dir = max(checkpoint_dir.glob('*'), key=lambda x: x.stat().st_mtime)

# latest file
checkpoint_path = max(checkpoint_dir.glob('*.ckpt'), key=lambda x: x.stat().st_mtime)

dataset_module = MIMICIIIDataModule.load_from_checkpoint(
    checkpoint_path,
    stage='finetune',
    data_fraction=1,
    data_config=config.data_config,
)

dataloader = dataset_module.train_dataloader()
start_time = time.time()
for batch in tqdm(dataloader):
    len(batch)

end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time:.2f} seconds")
