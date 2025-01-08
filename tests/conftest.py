import pytest

from cellink._core.dummy_data import DUMMY_COVARIATES, sim_adata, sim_gdata


@pytest.fixture
def dummy_covariates():
    return DUMMY_COVARIATES


@pytest.fixture
def adata():
    return sim_adata()


@pytest.fixture
def gdata(adata):
    return sim_gdata(adata=adata)
