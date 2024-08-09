import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import jit
from torch.utils.data import Dataset as TorchDataset


@dataclass(frozen=True)
class DatasetConfig:
    path: str
    variables_dropout: float
    data_fraction: float = 1.0
    max_events: int = 880
    max_minute: int = 1440
    min_input_minutes: int = 720
    
    def __post_init__(self):
        assert self.path is not None, "Dataset path must be provided"
        assert os.path.exists(self.path), f"Dataset {self.path} does not exist"
        assert 0 <= self.variables_dropout <= 1, "Variables dropout must be in [0, 1] range"
        assert 0 < self.data_fraction <= 1, "Data fraction must be in (0, 1] range"
        assert self.max_events > 0, "Max events must be positive"
        assert self.max_minute > 0, "Max minute must be positive"
        assert self.min_input_minutes > 0, "Min input minutes must be positive"


# TODO: think what is the best place to convert the data to datatypes expected by the model

# @profile
class Dataset(TorchDataset):
    def __init__(self, logger, config: DatasetConfig):
        self.config = config
        self.logger = logger
        events = pd.read_parquet(
            os.path.join(config.path, 'events.parquet'),
            columns=['stay_id', 'value', 'variable', 'minute']
        ) \
            .astype({'variable': 'category'}) \
            .set_index('stay_id')
        
        self.max_events = np.int64(config.max_events)
        self.max_minute = np.float32(config.max_minute)
        self.min_input_minutes = np.int64(config.min_input_minutes)
        
        # TODO: separate the demographics data from the events data
        self.variables_type = events['variable'].dtype
        self.num_variables = np.int16(len(self.variables_type.categories))
        self.drop_num = np.int16(self.num_variables * self.config.variables_dropout)
        
        events['variable'] = events['variable'].cat.codes.astype('int32')
        value_mean, value_std = events['value'].agg(['mean', 'std'])
        events['value'] = ((events['value'] - value_mean) / value_std).astype('float32')
        
        demographics = pd.read_parquet(
            os.path.join(config.path, 'demographics.parquet'),
            columns=['stay_id', 'variable', 'value']
        ) \
            .astype({'variable': 'category', 'value': 'float32'})
        age_idx = demographics['variable'] == 'Age'
        age_mean, age_std = demographics.loc[age_idx, 'value'].agg(['mean', 'std'])
        demographics.loc[age_idx, 'value'] = (
            (demographics.loc[age_idx, 'value'] - age_mean) / age_std).astype('float32')
        demographics = demographics.set_index(['stay_id', 'variable'])
        self.num_demographics = len(demographics.index.levels[1])
        self.data = {}
        
        # TODO: select data_fraction
        for stay_id, stay_events in events.groupby('stay_id'):
            timestamps = stay_events['minute'].unique()
            stay_events.sort_values('minute', inplace=True)
            stay_data = {
                # 'events': stay_events.astype({'minute': 'float32'}),
                'variable': stay_events['variable'].to_numpy(),
                'minute': stay_events['minute'].to_numpy(dtype='float32'),
                'value': stay_events['value'].to_numpy(),
                'demographics': demographics.loc[
                    [(stay_id, 'Age'), (stay_id, 'Gender')], 'value'].to_numpy(),
                # all unique minutes more than 720 and less than max
                # possible values for prediction window start time
                't1s': timestamps[
                    (timestamps >= self.min_input_minutes) & (timestamps < timestamps.max())],
            }
            if len(stay_data['t1s']) > 0:
                self.data[stay_id] = stay_data
        
        # if labels are required
        # mortality_labels = pd.read_parquet(
        #     os.path.join(config.path, 'mortality_labels.parquet'),
        # ).set_index('stay_id')
        #
        # for stay_id, stay_label in mortality_labels.iterrows():
        #     self.data[stay_id]['died'] = stay_label['died']
        
        self.stay_ids = list(self.data.keys())
        self.len = len(self.data)
    
    def __len__(self):
        return self.len
    
    def _select_window(
        self,
        minutes: np.ndarray,
        variables: np.ndarray,
        values: np.ndarray,
        t0,
        t1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        end_idx = minutes.searchsorted(t1, 'right')
        start_idx = max(minutes.searchsorted(t0, 'left'), end_idx - self.max_events)
        
        minutes = minutes[start_idx:end_idx]
        variables = variables[start_idx:end_idx]
        values = values[start_idx:end_idx]
        
        return minutes, variables, values
    
    def _drop_variables(
        self,
        minutes: np.ndarray,
        variables: np.ndarray,
        values: np.ndarray,
        drop_num: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        variables_to_drop = np.random.random_integers(0, self.num_variables, drop_num)
        
        idx = np.where(~np.isin(variables, variables_to_drop))
        
        if len(idx[0]) == 0:
            return None
        
        return minutes[idx], variables[idx], values[idx]
    
    @staticmethod
    @jit
    def _get_mean_values(values: np.ndarray, variables: np.ndarray, num_variables: int):
        sum_values = np.zeros(num_variables, dtype=values.dtype)
        count_values = np.zeros(num_variables, dtype=np.int16)
        for value, variable in zip(values, variables):
            sum_values[variable] += value
            count_values[variable] += 1
        
        mask = count_values > 0
        mean_values = np.where(mask, sum_values / count_values, 0)
        return mean_values, mask
    
    def _get_forecast_data(
        self,
        minutes: np.ndarray, variables: np.ndarray, values: np.ndarray,
        t1, t2
    ):
        start_idx = minutes.searchsorted(t1, 'right')
        end_idx = minutes.searchsorted(t2, 'right')
        
        variables = variables[start_idx:end_idx]
        values = values[start_idx:end_idx]
        
        forecast_values, forecast_mask = self._get_mean_values(
            values,
            variables,
            self.num_variables
        )
        
        return forecast_values, forecast_mask
    
    # @profile
    def __getitem__(self, idx):
        stay_id = self.stay_ids[idx]
        sample = self.data[stay_id]
        
        # TODO: check if it will work as well if we randomly select t1 from minutes index
        
        window = None
        while window is None:
            t1 = np.random.choice(sample['t1s'])
            t0 = t1 - 24 * 60  # 24 hours
            
            window = self._select_window(
                sample['minute'], sample['variable'], sample['value'], t0, t1
            )
            
            window = self._drop_variables(*window, self.drop_num)
        
        input_minutes, input_variables, input_values = window
        # normalize time to [-1,1] range
        input_times = (input_minutes - input_minutes[0]) / self.max_minute * 2 - 1
        
        t2 = t1 + 120  # prediction window is 2 hrs
        forecast_values, forecast_mask = self._get_forecast_data(
            sample['minute'], sample['variable'], sample['value'], t1, t2
        )
        
        return {
            'values': input_values,
            'times': input_times,
            'variables': input_variables,
            'demographics': sample['demographics'],
            'forecast_values': forecast_values,
            'forecast_mask': forecast_mask
        }
