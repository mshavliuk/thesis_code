import pytest

from workflow.scripts.latex_data_job import LatexDataJob


@pytest.fixture(scope="function", name='obj')
def latex_data_job_shortcut(latex_data_job: LatexDataJob):
    return latex_data_job


class TestLatexDataJob:
    def test_run(self, obj: LatexDataJob):
        obj.run()
