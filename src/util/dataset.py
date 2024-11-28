import itertools
import os
from pathlib import Path
from typing import (
    ClassVar,
    Literal,
)

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from numba import jit
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    field_serializer,
    field_validator,
)
from torch.utils.data import Dataset as TorchDataset

import src.util.variable_scalers as scaler_module
from src.models.strats_original import FeaturesInfo
from src.util.variable_scalers import (
    AbstractScaler,
    DemographicScaler,
    TimeScaler,
)

SUPPORTED_SCALERS = (
    scaler_module.VariableStandardScaler.__name__,
    scaler_module.VariableECDFScaler.__name__,
    scaler_module.UInt8ECDFScaler.__name__,
)


class DatasetConfig(BaseModel, extra='forbid'):
    base_dir: ClassVar = Path(os.environ.get('DATA_DIR', os.getcwd())).resolve()
    
    path: DirectoryPath = Field(description="Path to the dataset directory")
    variables_dropout: float = Field(
        ge=0, le=1, description="Fraction of variables to drop from the sample")
    max_events: int = Field(gt=0, description="Maximum number of events in the input window")
    max_minute: int = Field(gt=0, description="Maximum duration of the input window in minutes")
    
    scaler_class: Literal[SUPPORTED_SCALERS] = Field(description="Variable scaler class name")
    
    select_top: int = Field(
        ge=0, default=0, description="Number of top variables to select, 0 to select all")
    
    @field_validator('path', mode='before')
    @classmethod
    def relative_path(cls, path: str) -> str | Path:
        if not os.path.isabs(path):
            return cls.base_dir / path
        return path
    
    @field_serializer('path', mode='plain')
    @classmethod
    def serialize_path(cls, path: Path) -> str:
        try:
            # Check if path starts with base_dir; if so, return a relative path
            relative_path = path.relative_to(cls.base_dir)
            return str(relative_path)
        except ValueError:
            # If the path is not within base_dir, return it as an absolute path
            return str(path)

class FinetuneDatasetConfig(DatasetConfig, extra='ignore'):
    pass

class PretrainDatasetConfig(DatasetConfig, extra='forbid'):
    prediction_window: int = Field(
        ge=1, description="The size (in minutes) of the prediction window")
    min_input_minutes: int = Field(ge=1, description="Smallest input window size in minutes")


class AbstractDataset(TorchDataset):
    def __init__(self, config: DatasetConfig):
        self.max_events = config.max_events
        self.max_minute = config.max_minute
        self.config = config
        self.data = None
        self.rng = np.random.default_rng(42)
        
        events_pq = ds.dataset(os.path.join(config.path, 'events.parquet'), format='parquet')
        scanner = events_pq.scanner(columns=['variable'])
        self.num_variables = np.int16(pc.count_distinct(scanner.to_table()['variable']).as_py())
        if self.config.select_top > 0:
            self.num_variables = min(
                self.num_variables,
                self.config.select_top,
            )
        demographic_pq = ds.dataset(
            os.path.join(config.path, 'demographics.parquet'), format='parquet')
        self.num_demographics = np.int16(len(set(demographic_pq.schema.names) - {'stay_id'}))
        
        self.variable_scaler: AbstractScaler | None = None
        self.demographic_scaler: DemographicScaler | None = None
        self.time_scaler: TimeScaler | None = None
    
    def __len__(self):
        return len(self.data)
    
    @property
    def indexes(self):
        return np.arange(len(self.data))
    
    def get_scalers(self):
        return {
            'variable_scaler': self.variable_scaler,
            'demographic_scaler': self.demographic_scaler,
            'time_scaler': self.time_scaler
        }
    
    def set_scalers(self, scalers: dict):
        if self.data is not None:
            raise ValueError("Scalers have to be copied before loading data")
        self.variable_scaler: AbstractScaler = scalers['variable_scaler']
        self.demographic_scaler = scalers['demographic_scaler']
        self.time_scaler = scalers['time_scaler']
        self.num_variables = self.variable_scaler.num_variables
    
    def get_features_info(self) -> FeaturesInfo:
        return FeaturesInfo(
            features_num=self.num_variables,
            demographics_num=self.num_demographics,
        )
    
    def load_data(self):
        raise NotImplementedError
    
    def _drop_variables(
        self,
        minutes: np.ndarray,
        variables: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        keep_vars = self.rng.random(self.num_variables) > self.config.variables_dropout
        idx = np.where(keep_vars[variables])[0]
        return minutes[idx], variables[idx], values[idx]
    
    def _select_window(
        self,
        minutes: np.ndarray,
        variables: np.ndarray,
        values: np.ndarray,
        t0,
        t1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        end_idx = minutes.searchsorted(t1, 'right')
        start_idx = minutes.searchsorted(t0, 'left')
        minutes = minutes[start_idx:end_idx]
        variables = variables[start_idx:end_idx]
        values = values[start_idx:end_idx]
        
        return minutes, variables, values


class MemDataset(TorchDataset):
    """
    A simple in-memory dataset. Should only be used for val and test datasets.
    """
    
    def __init__(self, dataset: TorchDataset):
        # sort by length of values to reduce padding size during collation
        # use reverse order to have longest sequences first, so the memory is allocated only once
        self.data = sorted([dataset[i] for i in range(len(dataset))],
                           key=lambda x: len(x['values']), reverse=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class PretrainDataset(AbstractDataset):
    def __init__(self, config: PretrainDatasetConfig):
        super().__init__(config)
        self.config = config
    
    def load_data(self):
        events = pd.read_parquet(
            os.path.join(self.config.path, 'events.parquet'),
            columns=['stay_id', 'value', 'variable', 'minute'],
        )
        if self.config.select_top > 0:
            top_variables = events['variable'].value_counts().nlargest(self.config.select_top).index
            events = events[events['variable'].isin(top_variables)]
        
        if self.variable_scaler is None:
            events = events.astype({'variable': 'category'})
            variable_type = events['variable'].dtype
            VariableScaler = getattr(scaler_module, self.config.scaler_class)
            self.variable_scaler = VariableScaler().fit(events)
            self.time_scaler = TimeScaler(self.config.max_minute)
        else:
            variable_type = pd.CategoricalDtype(categories=self.variable_scaler.variable_categories)
            events = events.astype({'variable': variable_type})
        
        demographics = pd.read_parquet(os.path.join(self.config.path, 'demographics.parquet')) \
            .set_index('stay_id')
        
        if self.demographic_scaler is None:
            self.demographic_scaler = DemographicScaler()
            self.demographic_scaler.fit(demographics)
        
        events = self.variable_scaler.transform(events)
        events['variable'] = events['variable'].cat.codes.astype('int32')
        events.sort_values('minute', inplace=True)
        
        assert self.num_variables == len(variable_type.categories)
        
        demographics = self.demographic_scaler.transform(demographics).astype('float32')
        assert self.num_demographics == demographics.columns.size
        
        self.data = {}
        for stay_id, stay_events in events.groupby('stay_id', sort=False):
            minutes = stay_events['minute'].to_numpy()
            minutes -= minutes[0]
            
            timestamps = np.unique(minutes)
            # >= min_input_minutes
            min_t1_idx = timestamps.searchsorted(self.config.min_input_minutes, 'left')
            # < max_minute
            max_t1_idx = timestamps.searchsorted(minutes[-1], 'right') - 1
            
            t1s = timestamps[min_t1_idx:max_t1_idx]
            
            if len(t1s) == 0:
                continue
            
            self.rng.shuffle(t1s)
            t1s = itertools.cycle(t1s)
            
            self.data[stay_id] = {
                'variable': stay_events['variable'].to_numpy(),
                'minute': minutes,
                'value': stay_events['value'].to_numpy(),
                'demographics': demographics.loc[stay_id].to_numpy(),
                # all unique minutes more than 720 and less than max
                # possible values for prediction window start time - pred_gap
                't1s': t1s,
            }
        
        self.stay_ids = list(self.data.keys())
    
    @staticmethod
    @jit
    def _get_mean_values(values: np.ndarray, variables: np.ndarray, num_variables: int):
        sum_values = np.zeros(num_variables, dtype=values.dtype)
        count_values = np.zeros(num_variables, dtype=np.int16)
        for value, variable in zip(values, variables):
            sum_values[variable] += value
            count_values[variable] += 1
        
        mask = (count_values > 0).astype(np.bool)
        mean_values = np.where(mask, sum_values / count_values, 0).astype(values.dtype)
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
    
    def __getitem__(self, idx):
        if self.data is None:
            raise ValueError("Data has not been loaded")
        
        stay_id = self.stay_ids[idx]
        sample = self.data[stay_id]
        
        window, t1 = None, None
        while window is None or len(window[0]) == 0:
            t1 = next(sample['t1s'])  # end of input window
            t0 = t1 - self.max_minute
            
            window = self._select_window(
                sample['minute'], sample['variable'], sample['value'], t0, t1
            )
            if self.config.variables_dropout > 0:
                window = self._drop_variables(*window)
        
        minutes, variables, values = window
        # normalize time to [-1,1] range
        times = self.time_scaler.transform(minutes).astype(np.float32)
        
        t2 = t1 + self.config.prediction_window
        forecast_values, forecast_mask = self._get_forecast_data(
            sample['minute'], sample['variable'], sample['value'].astype(np.float32), t1, t2
        )
        forecast_values /= 256
        
        if (num_samples := len(values)) <= self.max_events:
            return {
                'values': values,
                'times': times,
                'variables': variables,
                'demographics': sample['demographics'],
                'forecast_values': forecast_values,
                'forecast_mask': forecast_mask
            }
        else:
            selected_events = self.rng.choice(num_samples, size=self.max_events, replace=False)
            return {
                'values': values[selected_events],
                'times': times[selected_events],
                'variables': variables[selected_events],
                'demographics': sample['demographics'],
                'forecast_values': forecast_values,
                'forecast_mask': forecast_mask
            }


class FinetuneDataset(AbstractDataset):
    def load_data(self):
        if None in self.get_scalers().values():
            raise ValueError("Scalers must be set before loading data")
        
        variables_type = pd.CategoricalDtype(categories=self.variable_scaler.variable_categories)
        
        mortality_labels = pd.read_parquet(
            self.config.path / 'labels.parquet',
        ).set_index('stay_id').astype({'died': 'float32'}) # convert to float32 to match the model output
        
        events = pd.read_parquet(
            os.path.join(self.config.path, 'events.parquet'),
            columns=['stay_id', 'value', 'variable', 'minute'],
            filters=[
                ('stay_id', 'in', mortality_labels.index),
                ('variable', 'in', self.variable_scaler.variable_categories)
            ]) \
            .astype({'variable': variables_type}) \
            .set_index('stay_id')
        
        events = self.variable_scaler.transform(events)
        events['variable'] = events['variable'].cat.codes.astype('int32')
        events.sort_values('minute', inplace=True)
        
        demographics = pd.read_parquet(os.path.join(self.config.path, 'demographics.parquet')) \
            .set_index('stay_id')
        # TODO: sort columns to ensure consistent order among different splits
        demographics = self.demographic_scaler.transform(demographics).astype('float32')
        self.num_demographics = demographics.columns.size
        
        self.data = {}
        
        for stay_id, stay_events in events.groupby('stay_id', sort=False):
            minutes = stay_events['minute'].to_numpy()
            minutes = minutes - minutes[0]
            slice_to = minutes.searchsorted(self.config.max_minute, 'right')
            
            self.data[stay_id] = {
                'variables': stay_events['variable'].to_numpy()[:slice_to],
                'minutes': minutes[:slice_to],
                'values': stay_events['value'].to_numpy()[:slice_to],
                'demographics': demographics.loc[stay_id].to_numpy(),
                # use array for label to simplify collation
                'label': np.array([mortality_labels.loc[stay_id, 'died']])
            }
        self.stay_ids = list(self.data.keys())
    
    def __getitem__(self, idx) -> dict:
        if self.data is None:
            raise ValueError("Data has not been loaded")
        
        stay_id = self.stay_ids[idx]
        stay = self.data[stay_id]
        
        triplets = None
        while triplets is None or len(triplets[0]) == 0:
            triplets = stay['minutes'], stay['variables'], stay['values']
            
            if self.config.variables_dropout > 0:
                triplets = self._drop_variables(*triplets)
        
        minutes, variables, values = triplets
        # normalize time to [-1,1] range, use the same scale as in pretrain dataset
        times = self.time_scaler.transform(minutes).astype(np.float32)
        
        if (num_samples := len(values)) <= self.max_events:
            return {
                'values': values,
                'times': times,
                'variables': variables,
                'label': stay['label'],
                'demographics': stay['demographics']
            }
        else:
            selected_events = self.rng.choice(num_samples, size=self.max_events, replace=False)
            return {
                'values': values[selected_events],
                'times': times[selected_events],
                'variables': variables[selected_events],
                'label': stay['label'],
                'demographics': stay['demographics']
            }
