import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest

from workflow.scripts.plotting_functions import get_plot_variables_distribution


class TestPlotVariablesDistribution:
    @pytest.fixture(scope='class')
    def df(self):
        return pd.read_csv('../tests/data/sbp_events.csv')
    
    @pytest.fixture(scope='class')
    def variables_data(self, df):
        codes = df['code'].unique()
        return pd.DataFrame({
            'code': codes,
            'min': [0] * len(codes),
            'max': [150] * len(codes),
            'variable': ['SBP'] * len(codes),
            'name': [
                'Manual BP [Systolic]',
                'Manual Blood Pressure Systolic Left',
            ]
        })
    
    def test_plot_and_save(self, df, variables_data):
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_and_save, schema = get_plot_variables_distribution(temp_dir)
            result = plot_and_save(('SBP',), df, variables_data)
            
            assert result.shape == (1, 3)
            assert result.iloc[0, 0] == "SBP"
            assert result.iloc[0, 1] == f"{temp_dir}/hist_SBP.eps"
    
    
    def test_plot_and_save_without_codes(self, df, variables_data):
        df['code'] = np.nan
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_and_save, schema = get_plot_variables_distribution(temp_dir)
            plot_and_save(('SBP',), df, variables_data)

    def test_plot_and_save_temperature(self):
        with open('/tmp/args_Temperature.pkl', 'rb') as f:
            variables, df, variables_data = pickle.load(f)
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_and_save, schema = get_plot_variables_distribution(temp_dir)
            result = plot_and_save(variables, df, variables_data)
            
            assert result.iloc[0, 0] == "Temperature"
            assert result.iloc[0, 1] == f"{temp_dir}/hist_Temperature.eps"
