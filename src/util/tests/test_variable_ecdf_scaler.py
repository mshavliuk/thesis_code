from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt

from src.util.variable_scalers import VariableECDFScaler


class TestVariableECDFScaler:
    @pytest.fixture(scope='function')
    def scaler(self):
        return VariableECDFScaler()
    
    @pytest.fixture
    def events(self, data_path: Path):
        return pd.read_parquet(
            data_path / 'events.parquet',
            columns=['stay_id', 'value', 'variable', 'minute'],
        ).astype({'variable': 'category', 'value': 'float32'})

    @pytest.fixture
    def large_events(self):
        return pd.read_parquet(
            '/home/user/.cache/thesis/data/original_strats_data/train/events.parquet',
            columns=['stay_id', 'value', 'variable', 'minute'],
        ).astype({'variable': 'category', 'value': 'float32'})
    
    def test_fit_scaler(self, events, scaler):
        scaler.fit(events)
        assert hasattr(scaler, 'ecdf')  # Check if ecdf attribute is set

    def test_transform_scaler(self, events, scaler):
        scaler.fit(events)
        scaled = scaler.transform(events)
        npt.assert_allclose(scaled['value'].min(), 0, atol=1e-3)
        npt.assert_allclose(scaled['value'].max(), 1, atol=1e-3)
        assert scaled['minute'].equals(events['minute'])
        assert scaled['stay_id'].equals(events['stay_id'])
        assert scaled['variable'].equals(events['variable'])
        assert scaled['value'].dtype == events['value'].dtype

    def test_inverse_transform_scaler(self, events, scaler):
        scaler.fit(events)
        scaled = scaler.transform(events)
        unscaled = scaler.inverse_transform(scaled)
        npt.assert_allclose(unscaled['value'], events['value'], atol=1e-3, rtol=1e-3)
        assert unscaled['minute'].equals(events['minute'])
        assert unscaled['stay_id'].equals(events['stay_id'])
        assert unscaled['variable'].equals(events['variable'])
        assert unscaled['value'].dtype == events['value'].dtype
    
    def test_interpolation_unseen_values(self, events, scaler):
        scaler.fit(events)
        unseen_events = events.copy()
        # set to random values
        unseen_events['value'] = np.random.uniform(-100, 100, len(unseen_events))
        scaled = scaler.transform(unseen_events)
        npt.assert_allclose(scaled['value'].min(), 0, atol=1e-3)
        npt.assert_allclose(scaled['value'].max(), 1, atol=1e-3)
    
    def test_inverse_transform_edge_cases(self, events, scaler):
        scaler.fit(events)
        scaled = scaler.transform(events)
        scaled.loc[0, 'value'] = -0.1  # Set an out-of-bound value
        scaled.loc[1, 'value'] = 1.1  # Set another out-of-bound value
        unscaled = scaler.inverse_transform(scaled)
        assert unscaled['value'].between(events['value'].min(), events['value'].max()).all()
        
    def test_double_transform_preserves_values(self, events: pd.DataFrame, scaler: VariableECDFScaler):
        events: pd.DataFrame = events.copy()
        events['value'] = np.random.uniform(-100, 100, len(events))
        # make sure every variable has a full range of values
        # append a pair of values to the end of the dataframe per each variable
        for variable in events['variable'].unique():
            additional_rows = pd.DataFrame({
                'variable': [variable, variable],
                'value': [-100, 100],
                'stay_id': [1, 2],
                'minute': [1, 2],
            }).astype(events.dtypes.to_dict())
            events = pd.concat([events, additional_rows], ignore_index=True)
        
        scaler.fit(events)
        
        unseen_events = events.copy()
        unseen_events['value'] = np.random.uniform(-100, 100, len(unseen_events))
        
        scaled = scaler.transform(unseen_events)
        unscaled = scaler.inverse_transform(scaled)
        npt.assert_allclose(unscaled['value'], unseen_events['value'], atol=1e-6, rtol=0.01)
