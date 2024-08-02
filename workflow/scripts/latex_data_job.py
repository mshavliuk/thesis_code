import os

import pandas as pd

from workflow.scripts.config import Config
from workflow.scripts.statistics_job import StatisticsJob


class LatexDataJob:
    outputs = {
        'variables_statistics': f'{Config.data_dir}/latex/variables_statistics.tex',
        'other_statistics': f'{Config.data_dir}/latex/other_statistics.tex',
        'raw_dataset_statistics': f'{Config.data_dir}/latex/raw_dataset_statistics.tex',
    }
    
    def __init__(self):
        os.makedirs(f'{Config.data_dir}/latex', exist_ok=True)
    
    def run(self):
        variables_statistics = pd.read_csv(StatisticsJob.outputs['variables'])
        self.process_variables_statistics(variables_statistics)
        self.process_other_statistics()
        raw_dataset_statistics = pd.read_csv(StatisticsJob.outputs['raw_dataset'])
        self.process_raw_dataset_statistics(raw_dataset_statistics)
    
    def process_variables_statistics(self, df):
        col_map = {
            'variable': 'Variable',
            'mean': 'Mean',
            'std': 'Std',
            'p0.01': '1%',
            'p0.5': '50%',
            'p0.99': '99%',
            'count': 'Count',
        }
        df = df.rename(columns=col_map)[list(col_map.values())]
        
        float_format = lambda x: f'{x:.2f}'
        int_format = lambda x: f'{x:,d}'
        noop = lambda x: x
        df.to_latex(
            self.outputs['variables_statistics'],
            index=False, escape=True, longtable=True,
            formatters=[noop] + [float_format] * 5 + [int_format], column_format='lcccccc',
            label='tab:variable_statistics', caption='Variable statistics')
    
    def process_other_statistics(self):
        features = pd.read_csv(StatisticsJob.outputs['features'], dtype={'value': object})
        patients = pd.read_csv(StatisticsJob.outputs['patient_journey'], dtype={'value': object})
        training = pd.read_csv(StatisticsJob.outputs['training'], dtype={'value': object})
        all = pd.concat([features, patients, training])
        
        with open(self.outputs['other_statistics'], 'w') as f:
            for idx, (key, value) in all.iterrows():
                f.write(f'\gdef\{key}{{{value}}}\n')
    
    def process_raw_dataset_statistics(self, df):
        int_format = lambda x: f'{x:,d}'
        bold_format = lambda x: f'\\textbf{{{x}}}'
        float_format = lambda x: f'{x:.2f}'
        df = df.set_index('file').astype(object)
        
        # add TOTAL row
        df.loc['Total'] = df.sum()
        df['size_mb'] = df['size_mb'].apply(float_format)
        df['rows'] = df['rows'].apply(int_format)
        df.reset_index(inplace=True)
        df.iloc[-1] = df.iloc[-1].apply(bold_format)
        df.columns = ['Table', 'Size (Mb)', 'Number of rows']
        
        # Escape underscores for latex
        # Has to be done manually due to escape=False, which is needed for bold formatting
        df['Table'] = df['Table'].apply(lambda x: x.replace('_', r'\_'))
        
        df.to_latex(self.outputs['raw_dataset_statistics'], index=False, escape=False)
