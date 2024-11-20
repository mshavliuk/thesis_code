import pytest
from pyspark.sql import (
    functions as F,
)

from workflow.scripts.data_extractor import DataExtractor


class TestDataExtractor:
    all_methods = [
        prop for prop in dir(DataExtractor)
        if callable(getattr(DataExtractor, prop)) and not prop.startswith("_")
    ]
    
    @pytest.mark.parametrize("method_name", all_methods)
    def test_read_methods(self, data_extractor: DataExtractor, method_name: str):
        method = getattr(data_extractor, method_name)
        result = method()
        assert result is not None
        assert result.first() is not None
    
    def test_read_weight_events(self, data_extractor: DataExtractor):
        weight_events = data_extractor.read_weight_events()
        assert set(weight_events.columns) == {'stay_id', 'time', 'value', 'unit', 'code',
                                              'admission_id', 'patient_id'}
    
    def test_read_icustays(self, data_extractor: DataExtractor):
        icustays = data_extractor.read_icustays()
        assert set(icustays.columns) == {'stay_id', 'time_start', 'time_end', 'admission_id',
                                         'patient_id', 'age'}
    
    def test_read_admissions(self, data_extractor: DataExtractor):
        admissions = data_extractor.read_admissions()
        assert set(admissions.columns) == {'admission_id', 'death_time', 'patient_id', 'died'}
    
    def test_read_items(self, data_extractor: DataExtractor):
        items = data_extractor.read_items()
        assert set(items.columns) == {'name', 'code'}
    
    def test_read_patients(self, data_extractor: DataExtractor):
        patients = data_extractor.read_patients()
        assert set(patients.columns) == {'patient_id', 'dob', 'gender'}
    
    def test_read_chartevents(self, data_extractor: DataExtractor):
        chartevents = data_extractor.read_chartevents()
        assert set(chartevents.columns) == {'stay_id', 'time', 'code', 'value', 'unit',
                                            'admission_id', 'patient_id'}
        
        assert chartevents.schema['value'].dataType.typeName() == 'string'
    
    def test_chartevents_have_categorical_values(self, data_extractor):
        chartevents = data_extractor.read_chartevents()
        result = chartevents.filter(F.col('code') == 115).first()  # 'CRR', categorical
        assert result is not None
        with pytest.raises(ValueError):
            float(result['value'])
    
    def test_chartevents_have_numerical_values(self, data_extractor):
        chartevents = data_extractor.read_chartevents()
        result = chartevents.filter(F.col('code') == 211).first()  # 'HR', numerical
        assert result is not None
        assert float(result['value'])
    
    def test_labevents_have_numerical_values(self, data_extractor):
        labevents = data_extractor.read_labevents()
        result = labevents.filter(F.col('code') == 51146).first()  # 'Basophils', numerical
        assert result is not None
        assert float(result['value'])
    
    def test_labevents_have_categorical_values(self, data_extractor):
        labevents = data_extractor.read_labevents()
        result = labevents.filter(F.col('code') == 50812).first()  # 'Intubated', categorical
        assert result is not None
        with pytest.raises(ValueError):
            float(result['value'])
