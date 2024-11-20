import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()
    input_path = Path(args.input_path).resolve()
    
    if not input_path.exists():
        raise FileNotFoundError(f'{input_path} not found')
    if not input_path.is_dir():
        raise NotADirectoryError(f'{input_path} is not a directory')
    
    output_path = Path('./src/util/tests/data').resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    labels = pd.read_parquet(input_path / 'labels.parquet').sample(frac=0.01)
    labels.to_parquet(output_path / 'labels.parquet')
    pd.read_parquet(
        input_path / 'events.parquet',
        filters=[('stay_id', 'in', labels['stay_id'])]
    ).to_parquet(output_path / 'events.parquet')
    pd.read_parquet(
        input_path / 'demographics.parquet',
        filters=[('stay_id', 'in', labels['stay_id'])]
    ).to_parquet(output_path / 'demographics.parquet')
    

if __name__ == '__main__':
    main()
