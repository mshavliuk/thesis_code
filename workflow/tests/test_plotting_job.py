import pytest
import tempfile

import pandas as pd
from pyspark.sql import (
    SparkSession,
    functions as F,
)

from workflow.scripts.data_processing_job import DataProcessingJob
from workflow.scripts.plotting_job import PlottingJob
from workflow.scripts.plotting_functions import get_plot_patient_journey
from workflow.scripts.statistics_job import StatisticsJob


@pytest.fixture(scope="function", name='obj')
def plotting_job_shortcut(plotting_job: PlottingJob):
    return plotting_job


class TestPlottingJob:
    @pytest.fixture(scope='module')
    def events(self, spark: SparkSession):
        return spark.read.parquet(DataProcessingJob.outputs['events'])
    
    @pytest.fixture(scope='module')
    def statistics(self):
        return pd.read_csv(StatisticsJob.outputs['variables'])
    
    def test_run(self, obj: PlottingJob):
        obj.run()

    def test_plot_patient_journey_tall(self, events, statistics):
        stay_id = 211944
        events = events.filter(F.col('stay_id') == stay_id).toPandas()
        # need source of events
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_func, schema = get_plot_patient_journey(temp_dir, statistics)
            plot_func((stay_id,), events)
    #
    # def test_plot_patient_journey_short(self, events, statistics):
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         plot_func, schema = get_plot_patient_journey(temp_dir, statistics)
    #         codes = events['code'].unique()[:10]
    #         events = events[events['code'].isin(codes)]
    #         plot_func((107032,), events)
