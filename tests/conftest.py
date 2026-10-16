import pandas as pd
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


@pytest.fixture
def pickled_parquet(monkeypatch):
    """Make ``pd.read_parquet`` read pickles, so parquet-reading code can be tested without pyarrow.

    Tests using this fixture write their inputs with ``DataFrame.to_pickle`` to the
    ``.parquet`` path the code under test expects.
    """

    def _read_pickle(path, columns=None):
        df = pd.read_pickle(path)
        return df if columns is None else df[columns]

    monkeypatch.setattr(pd, "read_parquet", _read_pickle)
