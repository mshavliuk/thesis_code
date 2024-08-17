import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from numba import jit
from torch.utils.data import Dataset as TorchDataset


@dataclass(frozen=True)
class DatasetConfig:
    path: str
    variables_dropout: float
    max_events: int
    max_minute: int
    min_input_minutes: int
    
    def __post_init__(self):
        assert self.path is not None, "Dataset path must be provided"
        assert os.path.exists(self.path), f"Dataset {self.path} does not exist"
        assert 0 <= self.variables_dropout <= 1, "Variables dropout must be in [0, 1] range"
        assert self.max_events > 0, "Max events must be positive"
        assert self.max_minute > 0, "Max minute must be positive"
        assert self.min_input_minutes > 0, "Min input minutes must be positive"


class VariablesMetadata:
    num_variables: int
    variable_categories: pd.CategoricalDtype


class AbstractScaler:
    def fit(self, data: pd.DataFrame):
        raise NotImplementedError
    
    def transform(self, data: pd.DataFrame):
        raise NotImplementedError
    
    def fit_transform(self, data: pd.DataFrame):
        self.fit(data)
        return self.transform(data)


class VariableScaler(AbstractScaler):
    variable_means: np.ndarray
    variable_stds: np.ndarray
    variable_categories: list[str]
    
    def fit(self, events: pd.DataFrame):
        self.variable_categories = events['variable'].cat.categories.to_list()
        variable_stats = (events.groupby('variable', observed=False)
                          .agg(mean=('value', 'mean'), std=('value', 'std'))
                          .sort_index())
        # some vars have only single value, so their std is 0.
        variable_stats.loc[variable_stats['std'].isin([float(0), float('nan')]), 'std'] = 1
        self.variable_means = variable_stats['mean'].to_numpy()
        self.variable_stds = variable_stats['std'].to_numpy()
        return self
    
    def transform(self, events: pd.DataFrame):
        return events.assign(value=(events['value'] - self.variable_means[
            events['variable'].cat.codes]) /
                                   self.variable_stds[events['variable'].cat.codes])
    
    def fit_transform(self, events: pd.DataFrame):
        self.fit(events)
        return self.transform(events)


# fixme: rename to DemographicScaler
class DemographicsScaler(AbstractScaler):
    age_mean: float
    age_std: float
    
    def fit(self, demographics: pd.DataFrame):
        self.age_mean, self.age_std = demographics['age'].agg(['mean', 'std'])
        return self
    
    def transform(self, demographics: pd.DataFrame):
        return demographics.assign(
            age=(demographics['age'] - self.age_mean) / self.age_std,
            gender=(demographics['gender'] * 2 - 1)
        )


class AbstractDataset(TorchDataset):
    num_variables: int
    num_demographics: int
    variable_scaler: VariableScaler = None
    demographic_scaler: DemographicsScaler = None
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
        self.max_events = np.int64(config.max_events)
        self.max_minute = np.float32(config.max_minute)
        self.min_input_minutes = np.int64(config.min_input_minutes)
        
        self.data = None
    
    def __len__(self):
        return len(self.data)
    
    def get_scalers(self):
        return {
            'variable_scaler': self.variable_scaler,
            'demographic_scaler': self.demographic_scaler
        }
    
    def set_scalers(self, scalers: dict):
        if self.data is not None:
            raise ValueError("Scalers have to be copied before loading data")
        self.variable_scaler = scalers['variable_scaler']
        self.demographic_scaler = scalers['demographic_scaler']
    
    def get_variables_metadata(self):
        return {
            'features_num': int(self.num_variables),
            'demographics_num': int(self.num_demographics)
        }
    
    def load_data(self):
        raise NotImplementedError

class PretrainDataset(AbstractDataset):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.data = None
        events_pq = ds.dataset(os.path.join(config.path, 'events.parquet'), format='parquet')
        scanner = events_pq.scanner(columns=['variable'])
        self.num_variables = np.int16(pc.count_distinct(scanner.to_table()['variable']).as_py())
        self.drop_num = np.int16(self.num_variables * self.config.variables_dropout)
        
        demographic_pq = ds.dataset(
            os.path.join(config.path, 'demographics.parquet'), format='parquet')
        self.num_demographics = np.int16(len(set(demographic_pq.schema.names) - {'stay_id'}))
    
    def load_data(self):
        events = pd.read_parquet(
            os.path.join(self.config.path, 'events.parquet'),
            columns=['stay_id', 'value', 'variable', 'minute']
        )
        
        if self.variable_scaler is None:
            events = events.astype({'variable': 'category'})
            variable_type = events['variable'].dtype
            self.variable_scaler = VariableScaler().fit(events)
        else:
            variable_type = pd.CategoricalDtype(categories=self.variable_scaler.variable_categories)
            events = events.astype({'variable': variable_type})
        
        events = self.variable_scaler.transform(events).astype({'value': 'float32'})
        events['variable'] = events['variable'].cat.codes.astype('int32')
        
        assert self.num_variables == len(variable_type.categories)
        
        demographics = pd.read_parquet(os.path.join(self.config.path, 'demographics.parquet')) \
            .set_index('stay_id')
        if self.demographic_scaler is None:
            self.demographic_scaler = DemographicsScaler()
            self.demographic_scaler.fit(demographics)
        
        demographics = self.demographic_scaler.transform(demographics).astype('float32')
        assert self.num_demographics == demographics.columns.size
        
        self.data = {}
        for stay_id, stay_events in events.groupby('stay_id'):
            timestamps = stay_events['minute'].unique()
            stay_events.sort_values('minute', inplace=True)
            stay_data = {
                'variable': stay_events['variable'].to_numpy(),
                'minute': stay_events['minute'].to_numpy(dtype='float32'),
                'value': stay_events['value'].to_numpy(),
                'demographics': demographics.loc[stay_id].to_numpy(),
                # all unique minutes more than 720 and less than max
                # possible values for prediction window start time
                't1s': timestamps[
                    (timestamps >= self.min_input_minutes) & (timestamps < timestamps.max())],
            }
            if len(stay_data['t1s']) > 0:
                self.data[stay_id] = stay_data
        
        self.stay_ids = list(self.data.keys())
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('data')
        return state
    
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
        stay_id = self.stay_ids[idx]
        sample = self.data[stay_id]
        
        # TODO: check if it will work as well if we randomly select t1 from minutes index
        
        window = None
        while window is None:
            t1 = np.random.choice(sample['t1s'])
            t0 = t1 - self.max_minute
            
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


class FinetuneDataset(AbstractDataset):
    def load_data(self):
        if self.variable_scaler is None or self.demographic_scaler is None:
            raise ValueError("Variable and demographic scalers must be set before loading data")
        
        variables_type = pd.CategoricalDtype(categories=self.variable_scaler.variable_categories)
        
        mortality_labels = pd.read_parquet(
            os.path.join(self.config.path, 'mortality_labels.parquet'),
        ).set_index('stay_id').astype({'died': 'float32'})
        #
        # if self.config.data_fraction < 1:
        #     torch.utils.data.Subset
        #     # TODO: extract into method and cover with tests
        #     fold_size = len(mortality_labels) * self.config.data_fraction
        #     start_idx = int(len(mortality_labels) / folds_number * fold_index)
        #     end_idx = int(start_idx + fold_size)
        #
        #     # Select the stay_ids for the current fold. Concat two stay_ids to allow cyclic indexing
        #     stay_ids = (mortality_labels.index.to_list() * 2)[start_idx:end_idx]
        #     mortality_labels = mortality_labels.loc[stay_ids]
        
        events = pd.read_parquet(
            os.path.join(self.config.path, 'events.parquet'),
            columns=['stay_id', 'value', 'variable', 'minute'],
            filters=[
                ('minute', '<=', self.max_minute),
                # only first 24h # TODO: try to randomly select 24h window, at least for training
                ('stay_id', 'in', mortality_labels.index)
            ]
        ) \
            .astype({'variable': variables_type})
        
        events = self.variable_scaler.transform(events).astype({'value': 'float32'})
        events['variable'] = events['variable'].cat.codes.astype('int32')
        events['time'] = (events['minute'] / self.max_minute * 2 - 1).astype('float32')
        
        demographics = pd.read_parquet(os.path.join(self.config.path, 'demographics.parquet')) \
            .set_index('stay_id')
        # TODO: sort columns to ensure consistent order
        demographics = self.demographic_scaler.transform(demographics).astype('float32')
        self.num_demographics = demographics.columns.size
        
        self.data = {}
        
        for stay_id, stay_events in events.groupby('stay_id'):
            stay_data = {
                'variables': stay_events['variable'].to_numpy(),
                'times': stay_events['time'].to_numpy(),
                'values': stay_events['value'].to_numpy(),
                'label': np.array([mortality_labels.loc[stay_id, 'died']]),
                'demographics': demographics.loc[stay_id].to_numpy()
            }
            self.data[stay_id] = stay_data
        self.stay_ids = list(self.data.keys())
    
    def __getitem__(self, idx):
        stay_id = self.stay_ids[idx]
        stay = self.data[stay_id]
        
        num_samples = len(stay['values'])
        
        if num_samples <= self.max_events:
            return stay
        else:
            selected_events = np.random.choice(num_samples, self.max_events, replace=False)
            return {
                'values': stay['values'][selected_events],
                'times': stay['times'][selected_events],
                'variables': stay['variables'][selected_events],
                'label': stay['label'],
                'demographics': stay['demographics']
            }
