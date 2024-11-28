import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


class AbstractScaler:
    variable_categories: list[str]
    
    def fit(self, data: pd.DataFrame):
        raise NotImplementedError
    
    def transform(self, data: pd.DataFrame, value_col: str = 'value'):
        raise NotImplementedError
    
    def fit_transform(self, data: pd.DataFrame):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: pd.DataFrame, value_col: str = 'value'):
        raise NotImplementedError
    
    @property
    def num_variables(self):
        raise NotImplementedError


class VariableStandardScaler(AbstractScaler):
    variable_means: np.ndarray
    variable_stds: np.ndarray
    
    def __eq__(self, other):
        return (
            isinstance(other, VariableStandardScaler)
            and np.array_equal(self.variable_means, other.variable_means)
            and np.array_equal(self.variable_stds, other.variable_stds)
            and self.variable_categories == other.variable_categories
        )
    
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
    
    def transform(self, events: pd.DataFrame, value_col: str = 'value'):
        stds = self.variable_stds[events['variable'].cat.codes]
        means = self.variable_means[events['variable'].cat.codes]
        return events.assign(**{value_col: (events[value_col] - means) / stds})
    
    def inverse_transform(self, data: pd.DataFrame, value_col: str = 'value'):
        stds = self.variable_stds[data['variable'].cat.codes]
        means = self.variable_means[data['variable'].cat.codes]
        return data.assign(**{value_col: data[value_col] * stds + means})
    
    @property
    def num_variables(self):
        return len(self.variable_categories)


class VariableECDFScaler(AbstractScaler):
    """
    A transformer that scales the values of each variable to their empirical cumulative distribution function (eCDF).
    The output range is [0, 1] for float32 and [0, 255] for uint8.
    """
    
    def __init__(self, dtype: np.dtypes.Float32DType | np.dtypes.UInt8DType = np.float32):
        self.ecdf = {}
        self.dtype = dtype
        self.inverse_dtype = None
    
    def fit(self, events: pd.DataFrame, value_col: str = 'value'):
        # WORKS MUCH BETTER IF events ARE ALREADY SORTED BY VALUE
        events = events.sort_values(value_col)
        self.inverse_dtype = events[value_col].dtype
        self.variable_categories = events['variable'].cat.categories.to_list()
        for variable, values in events.groupby('variable', observed=True)[value_col]:
            uniq_vals, cnt = np.unique(values.to_numpy(), return_counts=True)
            ranks = np.cumsum(cnt)
            
            if self.dtype == np.float32:
                ecdfs = ranks / len(values)
                # in the middle between adjacent values
                boundaries = uniq_vals
                self.ecdf[variable] = ecdfs, boundaries
            elif self.dtype == np.uint8:
                space = (ranks - cnt / 2) / len(values) * 256
                ecdfs = np.interp(np.linspace(0.5, 255.5, 256), space, uniq_vals)
                boundaries = np.interp(np.linspace(1, 255, 255), space, uniq_vals)
                self.ecdf[variable] = ecdfs, boundaries
        return self
    
    def _group_transformer(self, value_col: str, inverse=False):
        # if self.dtype == np.float32:
        #     def transform(group: pd.DataFrame):
        #         ecdfs, boundaries = self.ecdf[group.name]
        #         if inverse:
        #             group[value_col] = np.interp(
        #                 group[value_col].values, ecdfs, sorted_values)
        #         else:
        #             group[value_col] = np.interp(
        #                 group[value_col].values, sorted_values, sorted_quantiles)
        #         return group
        #
        #     return transform
        # elif self.dtype == np.uint8:
        def transform(group: pd.DataFrame):
            ecdfs, boundaries = self.ecdf[group.name]
            if inverse:
                if self.dtype == np.float32:
                    group[value_col] = np.interp(
                        group[value_col].values, ecdfs, boundaries)
                elif self.dtype == np.uint8:
                    # turn index into value
                    group[value_col] = ecdfs[group[value_col].values]
            else:
                # get the index with closest value
                if self.dtype == np.float32:
                    group[value_col] = np.interp(
                        group[value_col].values, boundaries, ecdfs)
                elif self.dtype == np.uint8:
                    sorted_group = group.sort_values(value_col)
                    group[value_col] = pd.Series(
                        np.searchsorted(boundaries, sorted_group[value_col].values),
                        index=sorted_group.index,
                        dtype=np.uint8
                    )
            return group
        
        return transform
    
    def transform(self, events: pd.DataFrame, value_col: str = 'value'):
        transform_group = self._group_transformer(inverse=False, value_col=value_col)
        return events.groupby('variable', observed=True, as_index=False, group_keys=False) \
            .progress_apply(transform_group).astype({value_col: self.dtype})
    
    def inverse_transform(self, events: pd.DataFrame, value_col: str = 'value'):
        inverse_transform_group = self._group_transformer(inverse=True, value_col=value_col)
        return events.groupby('variable', observed=True, as_index=False, group_keys=False) \
            .progress_apply(inverse_transform_group).astype({value_col: self.inverse_dtype})
    
    def __eq__(self, other):
        if not isinstance(other, VariableECDFScaler):
            return False
        if self.variable_categories != other.variable_categories:
            return False
        for var in self.variable_categories:
            if not np.array_equal(self.ecdf[var], other.ecdf[var]):
                return False
        return True
    
    @property
    def num_variables(self):
        return len(self.variable_categories)


class UInt8ECDFScaler(AbstractScaler):
    
    # TODO: best idea: use a single ECDF scaler class with a flag for uint8 scaling
    #   This could allow pretraining with float32 and finetuning with uint8 while using the same scaler
    
    """
    A transformer that scales the values of each variable to their empirical cumulative distribution function (eCDF).
    The output range is [0, 255]
    """
    
    def __init__(self):
        self.ecdf = {}
    
    def fit(self, events: pd.DataFrame):
        self.variable_categories = events['variable'].cat.categories.to_list()
        for variable, values in events.groupby('variable', observed=True)['value']:
            values = values.to_numpy()
            values.sort()
            
            space = np.linspace(0, 256, len(values))
            ecdfs = np.interp(np.linspace(0.5, 255.5, 256), space, values)
            boundaries = np.interp(np.linspace(1, 255, 255), space, values)
            self.ecdf[variable] = (ecdfs, boundaries)
        return self
    
    def _group_transformer(self, value_col: str, inverse=False):
        def transform(group: pd.DataFrame):
            ecdfs, boundaries = self.ecdf[group.name]
            if inverse:
                # turn index into value
                group[value_col] = ecdfs[group[value_col].values]
            else:
                # get the index with closest value
                group[value_col] = pd.Series(
                    np.searchsorted(boundaries, group[value_col].values),
                    index=group.index,
                    dtype=np.uint8
                )
            return group
        
        return transform
    
    def transform(self, events: pd.DataFrame, value_col: str = 'value'):
        transform_group = self._group_transformer(inverse=False, value_col=value_col)
        return events.groupby('variable', observed=True, as_index=False, group_keys=False) \
            .progress_apply(transform_group)
    
    def inverse_transform(self, events: pd.DataFrame, value_col: str = 'value'):
        inverse_transform_group = self._group_transformer(inverse=True, value_col=value_col)
        return events.groupby('variable', observed=True, as_index=False, group_keys=False) \
            .progress_apply(inverse_transform_group)
    
    def __eq__(self, other):
        if not isinstance(other, VariableECDFScaler):
            return False
        if self.variable_categories != other.variable_categories:
            return False
        for var in self.variable_categories:
            if not np.array_equal(self.ecdf[var], other.ecdf[var]):
                return False
        return True
    
    @property
    def num_variables(self):
        return len(self.ecdf)


class DemographicScaler(AbstractScaler):
    age_mean: float
    age_std: float
    
    def __eq__(self, other):
        return (
            isinstance(other, DemographicScaler)
            and self.age_mean == other.age_mean
            and self.age_std == other.age_std
        )
    
    def fit(self, demographics: pd.DataFrame):
        self.age_mean, self.age_std = demographics['age'].agg(['mean', 'std'])
        return self
    
    def transform(self, demographics: pd.DataFrame):
        return demographics.assign(
            age=(demographics['age'] - self.age_mean) / self.age_std,
            gender=(demographics['gender'] * 2 - 1)
        )


class TimeScaler:
    def __init__(self, max_minute: int):
        self.half_max_minute = np.float64(max_minute / 2)
    
    def transform(self, minutes: np.ndarray):
        return (minutes - (minutes.min() + self.half_max_minute)) / self.half_max_minute
    
    def __eq__(self, other):
        return isinstance(other, TimeScaler) and np.isclose(self.half_max_minute,
                                                            other.half_max_minute)
