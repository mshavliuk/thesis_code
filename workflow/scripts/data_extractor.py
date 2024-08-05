import logging
from functools import cache

from pyspark.sql import (
    DataFrame,
    SparkSession,
    functions as F,
)

from workflow.scripts.config import Config


class DataExtractor:
    """
    A class to directly read the MIMIC-III dataset parquet files or extract some data from them.
    
    All methods are pure functions, so they can be used in a distributed environment.
    """
    col_names_map = {
        'ICUSTAY_ID': 'stay_id',
        'ITEMID': 'code',
        'AMOUNT': 'value',
        'AMOUNTUOM': 'unit',
        'VALUE': 'value',
        'VALUENUM': 'value',
        'VALUEUOM': 'unit',
        'SUBJECT_ID': 'patient_id',
        'CHARTTIME': 'time',
        'STARTTIME': 'time_start',
        'ENDTIME': 'time_end',
        'HADM_ID': 'admission_id',
        'HOSPITAL_EXPIRE_FLAG': 'died',
        'DEATHTIME': 'death_time',
        'AGE': 'age',
        'DOB': 'dob',
        'GENDER': 'gender',
        'LABEL': 'name',
        'INTIME': 'time_start',
        'OUTTIME': 'time_end',
    }
    
    # some weight events are extracted from INPUTEVENTS table and do not have codes, so they are
    # hardcoded by one of the weight evnet codes
    WEIGHT_CODE = 224639
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _filter_not_null(self, df: DataFrame, column_names: set[str]) -> DataFrame:
        for col in column_names:
            col_type = df.schema[col].dataType
            df = df.filter(F.col(col).isNotNull())
            # update schema to non-nullable
            df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0).cast(col_type)))
        
        return df
    
    def _select_known_columns(self, df: DataFrame) -> DataFrame:
        known_columns = set(self.col_names_map.values())
        df_columns = set(df.columns)
        return df.select(*[col for col in df_columns if col in known_columns])
    
    @cache
    def read_patients(self) -> DataFrame:
        patients = self.spark.read.parquet(f'{Config.data_dir}/raw/PATIENTS.parquet')
        patients = self._filter_not_null(patients, {'SUBJECT_ID', 'DOB', 'GENDER'}) \
            .withColumnsRenamed(self.col_names_map)
        return self._select_known_columns(patients).cache()
    
    @cache
    def read_icustays(self) -> DataFrame:
        patients = self.read_patients()
        icustays = (self.spark.read.parquet(f'{Config.data_dir}/raw/ICUSTAYS.parquet'))
        icustays = self._filter_not_null(
            icustays, {'INTIME', 'OUTTIME', 'ICUSTAY_ID', 'HADM_ID', 'SUBJECT_ID'}) \
            .withColumnsRenamed(self.col_names_map) \
            .alias('icu') \
            .join(patients.select('dob', 'patient_id'), how='left', on='patient_id') \
            .withColumn('age', F.coalesce(F.year('time_start') - F.year('dob'), F.lit(0))) \
            .filter(F.col('age') >= 18) \
            .select('icu.*', 'age')
        return self._select_known_columns(icustays).cache()
    
    @cache
    def get_icustay_ids(self) -> DataFrame:
        return self.read_icustays().select('stay_id').distinct().cache()
    
    @cache
    def get_admission_ids(self) -> DataFrame:
        return self.read_admissions().select('admission_id').distinct().cache()
    
    @cache
    def read_admissions(self) -> DataFrame:
        admissions = self.spark.read.parquet(f'{Config.data_dir}/raw/ADMISSIONS.parquet')
        admissions = self._filter_not_null(
            admissions, {'HADM_ID', 'SUBJECT_ID', 'HOSPITAL_EXPIRE_FLAG'}) \
            .withColumnsRenamed(self.col_names_map) \
            .withColumn('died', F.col('died').cast('boolean')) \
            .alias('a') \
            .join(self.read_icustays(), on='admission_id', how='inner') \
            .select('a.*')
        
        return self._select_known_columns(admissions).cache()
    
    @cache
    def read_items(self) -> DataFrame:
        d_items = self.spark.read.parquet(f'{Config.data_dir}/raw/D_ITEMS.parquet')
        d_labitems = self.spark.read.parquet(f'{Config.data_dir}/raw/D_LABITEMS.parquet')
        
        all_items = d_items.unionByName(d_labitems, allowMissingColumns=True)
        all_items = self._filter_not_null(all_items, {'ITEMID', 'LABEL'}) \
            .withColumnsRenamed(self.col_names_map)
        return self._select_known_columns(all_items).cache()
    
    def read_outputevents(self) -> DataFrame:
        outputevents = self.spark.read.parquet(f'{Config.data_dir}/raw/OUTPUTEVENTS.parquet')
        outputevents = self._filter_not_null(
            outputevents, {'CHARTTIME', 'ITEMID', 'VALUE', 'ICUSTAY_ID', 'SUBJECT_ID'}) \
            .withColumnsRenamed(self.col_names_map) \
            .join(self.get_icustay_ids(), on='stay_id', how='inner')
        return self._select_known_columns(outputevents)
    
    def read_chartevents(self) -> DataFrame:
        # some chartevents are categorical so value is string
        chartevents = self.spark.read.parquet(f'{Config.data_dir}/raw/CHARTEVENTS.parquet') \
            .filter((F.col('ERROR') == 0) | F.col('ERROR').isNull()) \
            .filter((F.col('VALUENUM').isNotNull()) | F.col('VALUE').isNotNull())
        # some chartevents have nullable ICUSTAY_ID, which is still valid
        # RR my: 9,803,802
        # after joining icu 8,115,504
        
        chartevents = self._filter_not_null(
            chartevents, {'CHARTTIME', 'ITEMID', 'SUBJECT_ID', 'ICUSTAY_ID'})
        chartevents = chartevents \
            .withColumn('new_value', F.coalesce(F.col('VALUENUM'), F.col('VALUE'), F.lit('NONE'))) \
            .drop('VALUENUM', 'VALUE') \
            .withColumnRenamed('new_value', 'value') \
            .withColumnsRenamed(self.col_names_map) \
            .join(self.get_icustay_ids(), on='stay_id', how='inner')
        return self._select_known_columns(chartevents)
    # RR count 8,115,504
    
    def read_labevents(self) -> DataFrame:
        # intubated labevents are categorical so value is string
        labevents = self.spark.read.parquet(f'{Config.data_dir}/raw/LABEVENTS.parquet') \
            .filter((F.col('VALUENUM').isNotNull()) | F.col('VALUE').isNotNull())
        labevents = self._filter_not_null(
            labevents, {'HADM_ID', 'CHARTTIME', 'ITEMID', 'SUBJECT_ID'})
        labevents = labevents \
            .withColumn('new_value', F.coalesce(F.col('VALUENUM'), F.col('VALUE'), F.lit('NONE'))) \
            .drop('VALUENUM', 'VALUE') \
            .withColumnRenamed('new_value', 'value') \
            .withColumnsRenamed(self.col_names_map)
        return self._select_known_columns(labevents)
    
    def read_inputevents_mv(self) -> DataFrame:
        # FIXME: some inputevents have zero amount and are not useful
        inputevents = self.spark.read.parquet(f'{Config.data_dir}/raw/INPUTEVENTS_MV.parquet')
        inputevents = self._filter_not_null(
            inputevents,
            {'STARTTIME', 'ENDTIME', 'ITEMID', 'AMOUNT', 'ICUSTAY_ID', 'AMOUNTUOM', 'SUBJECT_ID'}) \
            .withColumnsRenamed(self.col_names_map) \
            .join(self.get_icustay_ids(), on='stay_id', how='inner')
        return self._select_known_columns(inputevents)
    
    def read_inputevents_cv(self) -> DataFrame:
        # FIXME: some inputevents have zero amount and are not useful
        inputevents = self.spark.read.parquet(f'{Config.data_dir}/raw/INPUTEVENTS_CV.parquet')
        inputevents = self._filter_not_null(
            inputevents, {'CHARTTIME', 'ITEMID', 'AMOUNT', 'ICUSTAY_ID', 'SUBJECT_ID'}) \
            .withColumnsRenamed(self.col_names_map) \
            .join(self.get_icustay_ids(), on='stay_id', how='inner')
        return self._select_known_columns(inputevents)
    
    # their total rows 17,527,935, my 17,527,935
    # my prefiltered by amount only 2,496,047, their 2,496,047
    # my with all filters 2,496,047
    
    def read_weight_events(self) -> DataFrame:
        inputevents = self.spark.read.parquet(f'{Config.data_dir}/raw/INPUTEVENTS_MV.parquet')
        inputevents = self._filter_not_null(
            inputevents, {'STARTTIME', 'PATIENTWEIGHT', 'ICUSTAY_ID', 'SUBJECT_ID'}) \
            .drop('ITEMID', 'AMOUNT', 'AMOUNTUOM', 'ENDTIME') \
            .withColumnsRenamed({
            'PATIENTWEIGHT': 'value',
            'STARTTIME': 'time'}) \
            .withColumnsRenamed(self.col_names_map) \
            .join(self.get_icustay_ids(), on='stay_id', how='inner') \
            .withColumn('code', F.lit(self.WEIGHT_CODE)) \
            .withColumn('unit', F.lit('kg'))
        return self._select_known_columns(inputevents)
