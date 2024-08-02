import pytest

from workflow.scripts.statistics_job import StatisticsJob


@pytest.fixture(scope="function", name='obj')
def statistics_job_shortcut(statistics_job: StatisticsJob):
    return statistics_job

class TestStatisticsJob:
    def test_run(self, obj: StatisticsJob):
        obj.run()
