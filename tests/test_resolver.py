from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from cellink import DonorData
from cellink._core.dummy_data import sim_adata, sim_gdata
from cellink.at.base_model import BaseModel
from cellink.at.gwas import GWAS
from cellink.at.resolver import get_model_matrix
from cellink.at.structlmm import StructLMM


@pytest.fixture
def dd():
    donordata = DonorData(G=sim_gdata(), C=sim_adata())
    rng = np.random.default_rng(0)
    donordata.G.obs["age"] = rng.integers(20, 60, donordata.G.n_obs).astype(float)
    donordata.G.obs["phenotype"] = rng.standard_normal(donordata.G.n_obs)
    return donordata


def test_get_model_matrix_donor_only_formula(dd):
    """A formula referencing only donor-level variables must not crash on an empty cell-level dict."""
    y = get_model_matrix(dd, "phenotype - 1", target_level="donor")
    assert y.shape == (dd.G.n_obs, 1)
    np.testing.assert_allclose(y.to_numpy().ravel(), dd.G.obs["phenotype"].to_numpy())


def test_get_model_matrix_cell_only_formula(dd):
    """Symmetric case: a cell-only formula must not crash on an empty donor-level dict."""
    x = get_model_matrix(dd, "celltype", target_level="cell")
    assert x.shape[0] == dd.C.n_obs


def test_get_model_matrix_donor_aggregation(dd):
    """dmax(<cell-level gene>) aggregates a cell-level variable up to one value per donor."""
    X = get_model_matrix(dd, "age + dmax(G0)", target_level="donor")
    assert X.shape == (dd.G.n_obs, 3)  # Intercept, age, dmax(G0)
    assert "dmax(G0)" in X.columns
    expected = dd.C[:, "G0"].X.ravel()
    expected = (
        pd.Series(np.asarray(expected).ravel(), index=dd.C.obs_names)
        .groupby(dd.C.obs[dd.donor_id], observed=True)
        .max()
    )
    np.testing.assert_allclose(X["dmax(G0)"].to_numpy(), expected.reindex(X.index).to_numpy())


def test_get_model_matrix_cell_aggregation_crepeat(dd):
    """crepeat(<donor-level var>) broadcasts a donor-level variable down to every cell."""
    X = get_model_matrix(dd, "crepeat(age)", target_level="cell")
    assert X.shape == (dd.C.n_obs, 2)  # Intercept, crepeat(age)
    expected = dd.C.obs[dd.donor_id].map(dd.G.obs["age"]).to_numpy()
    np.testing.assert_allclose(X["crepeat(age)"].to_numpy(), expected)


def test_get_model_matrix_plain_anndata():
    adata = sim_adata()
    adata.obs["pheno"] = np.random.randn(adata.n_obs)
    y = get_model_matrix(adata, "pheno - 1")
    assert y.shape == (adata.n_obs, 1)
    np.testing.assert_allclose(y.to_numpy().ravel(), adata.obs["pheno"].to_numpy())


def test_get_model_matrix_plain_anndata_rejects_donor_agg():
    adata = sim_adata()
    adata.obs["pheno"] = np.random.randn(adata.n_obs)
    with pytest.raises(ValueError, match="not applicable for a single AnnData"):
        get_model_matrix(adata, "dmean(pheno) - 1")


def test_get_model_matrix_dataframe_passthrough():
    df = pd.DataFrame({"y": np.random.rand(20), "x1": np.random.rand(20)})
    y = get_model_matrix(df, "y - 1")
    np.testing.assert_allclose(y.to_numpy().ravel(), df["y"].to_numpy())


def test_gwas_data_equals_raw_numpy():
    """GWAS(Y=<formula>, data=...) must give the same result as GWAS(Y=<same values as ndarray>)."""
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "y": rng.standard_normal(n),
            "cov1": rng.standard_normal(n),
        }
    )
    g = rng.standard_normal((n, 5))

    gwas_raw = GWAS(Y=df[["y"]].to_numpy(), F=df[["cov1"]].assign(intercept=1.0)[["intercept", "cov1"]].to_numpy())
    gwas_raw.test_association(g)

    gwas_formula = GWAS(Y="y", F="cov1", data=df)
    gwas_formula.test_association(g)

    np.testing.assert_allclose(gwas_formula.getPv(), gwas_raw.getPv(), rtol=1e-8)
    np.testing.assert_allclose(gwas_formula.getBetaSNP(), gwas_raw.getBetaSNP(), rtol=1e-8)


def test_gwas_donordata_formula_runs(dd):
    """GWAS against a DonorData with formula strings runs end to end and gives finite p-values."""
    gwas = GWAS(Y="phenotype", F="age", data=dd, target_level="donor")
    g = np.random.default_rng(2).standard_normal((dd.G.n_obs, 4))
    gwas.test_association(g)
    pv = gwas.getPv()
    assert np.all(np.isfinite(pv))
    assert pv.shape == (4, 1)


def test_gwas_missing_data_raises_on_formula_string():
    with pytest.raises(ValueError, match="Mandatory to provide `data`"):
        GWAS(Y="phenotype")


def test_skat_data_resolver_multiple_variants():
    """Regression test: `Skat.run_test(data=..., X=...)` used to raise TypeError
    (`isinstance(X, str | list[str])` is invalid at runtime) and, even past that,
    the DonorData branch referenced a nonexistent `.donor_data` attribute. Both are
    fixed; this exercises all three data containers with both a single column name
    and a multi-column formula string for X."""
    pytest.importorskip("chiscore", reason="Skat needs chiscore, install with `conda install -c conda-forge chiscore`")
    from cellink.at.skat import Skat

    rng = np.random.default_rng(4)
    n = 500
    snp_cols = [f"snp{i}" for i in range(5)]
    snp_formula = " + ".join(snp_cols)
    skat = Skat(min_threshold=1)

    adata = sim_adata()
    n = adata.n_obs
    adata.obs["pheno"] = rng.standard_normal(n)
    adata.obs[snp_cols] = rng.integers(0, 3, size=(n, 5)).astype(float)

    pv_multi = skat.run_test(data=adata, Y="pheno", X=snp_formula)
    pv_single = skat.run_test(data=adata, Y="pheno", X=snp_cols[0])
    assert np.isfinite(pv_multi) and np.isfinite(pv_single)

    df = adata.obs[["pheno", *snp_cols]].copy()
    pv_df = skat.run_test(data=df, Y="pheno", X=snp_formula)
    np.testing.assert_allclose(pv_df, pv_multi)

    dd = DonorData(G=sim_gdata(), C=sim_adata())
    dd.G.obs["pheno"] = rng.standard_normal(dd.G.n_obs)
    dd.G.obs[snp_cols] = rng.integers(0, 3, size=(dd.G.n_obs, 5)).astype(float)
    pv_dd = skat.run_test(data=dd, Y="pheno", X=snp_formula)
    assert np.isfinite(pv_dd)


def test_skat_data_equals_raw_numpy():
    """Skat.run_test(data=...) must give the same result as the equivalent raw-numpy call,
    now that it goes through the same fetch_raw_slot resolver as GWAS/StructLMM."""
    pytest.importorskip("chiscore", reason="Skat needs chiscore, install with `conda install -c conda-forge chiscore`")
    from cellink.at.skat import Skat

    rng = np.random.default_rng(6)
    n = 500
    snp_cols = [f"snp{i}" for i in range(5)]
    df = pd.DataFrame({"pheno": rng.standard_normal(n)})
    df[snp_cols] = rng.integers(0, 3, size=(n, 5)).astype(float)

    skat = Skat(min_threshold=1)
    pv_formula = skat.run_test(data=df, Y="pheno", X=" + ".join(snp_cols))
    pv_raw = skat.run_test(Y=df[["pheno"]].to_numpy(), X=df[snp_cols].to_numpy())
    np.testing.assert_allclose(pv_formula, pv_raw)


def test_skat_donordata_defaults_to_donor_level():
    """Skat's variants are donor-level; a DonorData resolves against `.G` without target_level="donor"."""
    pytest.importorskip("chiscore", reason="Skat needs chiscore, install with `conda install -c conda-forge chiscore`")
    from cellink.at.skat import Skat

    rng = np.random.default_rng(7)
    snp_cols = [f"snp{i}" for i in range(5)]
    dd = DonorData(G=sim_gdata(), C=sim_adata())
    dd.G.obs["pheno"] = rng.standard_normal(dd.G.n_obs)
    dd.G.obs[snp_cols] = rng.integers(0, 3, size=(dd.G.n_obs, 5)).astype(float)

    skat = Skat(min_threshold=1)
    pv = skat.run_test(data=dd, Y="pheno", X=" + ".join(snp_cols))
    assert np.isfinite(pv)


def test_structlmm_data_equals_raw_numpy():
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame({"y": rng.standard_normal(n), "cov1": rng.standard_normal(n)})
    E = rng.standard_normal((n, 2))

    s_raw = StructLMM(y=df[["y"]].to_numpy(), E=E, F=df[["cov1"]].assign(intercept=1.0)[["intercept", "cov1"]].to_numpy())
    s_formula = StructLMM(y="y", E=E, F="cov1", data=df)

    np.testing.assert_allclose(s_formula.y, s_raw.y)
    np.testing.assert_allclose(s_formula.F, s_raw.F)


def test_get_model_matrix_family_operator_expansion():
    df = pd.DataFrame({f"X_{i}": np.arange(5) * i for i in range(1, 6)})
    X = get_model_matrix(df, "@X[1:3] - 1")
    assert list(X.columns) == ["X_1", "X_2", "X_3"]
    np.testing.assert_allclose(X["X_2"].to_numpy(), df["X_2"].to_numpy())


def test_get_model_matrix_family_operator_with_function_wrap():
    df = pd.DataFrame({f"X_{i}": np.arange(1, 6) * i for i in range(1, 3)})
    X = get_model_matrix(df, "@np.log1p(X[1:2]) - 1")
    assert list(X.columns) == ["np.log1p(X_1)", "np.log1p(X_2)"]
    np.testing.assert_allclose(X["np.log1p(X_1)"].to_numpy(), np.log1p(df["X_1"].to_numpy()))


@pytest.mark.parametrize("agg_func", ["dmean", "dmax", "dmedian", "dfirst"])
def test_get_model_matrix_donor_aggregation_functions(dd, agg_func):
    X = get_model_matrix(dd, f"{agg_func}(G0) - 1", target_level="donor")
    assert X.shape == (dd.G.n_obs, 1)
    assert X.columns[0] == f"{agg_func}(G0)"


def test_get_model_matrix_donordata_requires_target_level(dd):
    with pytest.raises(ValueError, match="target_level"):
        get_model_matrix(dd, "age - 1")


def test_get_model_matrix_rejects_crepeat_in_donor_formula(dd):
    with pytest.raises(ValueError, match="crepeat"):
        get_model_matrix(dd, "crepeat(age)", target_level="donor")


def test_get_model_matrix_rejects_donor_agg_in_cell_formula(dd):
    with pytest.raises(ValueError, match="Donor aggregation functions"):
        get_model_matrix(dd, "dmean(G0)", target_level="cell")


def test_get_model_matrix_dataframe_rejects_donor_agg():
    df = pd.DataFrame({"y": np.arange(5.0)})
    with pytest.raises(ValueError, match="not applicable for a pandas DataFrame"):
        get_model_matrix(df, "dmean(y) - 1")
    with pytest.raises(ValueError, match="not applicable for a pandas DataFrame"):
        get_model_matrix(df, "crepeat(y) - 1")


def test_get_model_matrix_missing_formulaic_raises_helpful_error(monkeypatch):
    _block_module(monkeypatch, "formulaic")
    df = pd.DataFrame({"y": np.arange(5.0)})
    with pytest.raises(ImportError, match="pip install cellink\\[at\\]"):
        get_model_matrix(df, "y - 1")


def _block_module(monkeypatch, name):
    """Simulate `name` being uninstalled, including any of its submodules already cached from earlier tests."""
    for mod_name in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
        monkeypatch.setitem(sys.modules, mod_name, None)


def test_skat_missing_limix_core_raises_helpful_error(monkeypatch):
    _block_module(monkeypatch, "limix_core")
    from cellink.at.skat import _skat_test

    with pytest.raises(ImportError, match="pip install limix-core"):
        _skat_test(np.zeros(5), np.zeros((5, 1)))


def test_structlmm_missing_limix_core_raises_helpful_error(monkeypatch):
    rng = np.random.default_rng(5)
    n = 30
    s = StructLMM(y=rng.standard_normal((n, 1)), E=rng.standard_normal((n, 2)), F=rng.standard_normal((n, 1)))

    _block_module(monkeypatch, "limix_core")
    with pytest.raises(ImportError, match="pip install limix-core"):
        s.interaction_test(rng.standard_normal((n, 3)))


def test_base_model_run_smoke():
    class _DummyModel(BaseModel):
        required_slots = ["Y", "F"]
        add_intercept_slots = ["F"]

        def _post_init(self):
            pass

        def _run(self, Y, F):
            return Y, F

    df = pd.DataFrame({"y": np.arange(5.0), "cov1": np.arange(5.0) * 2})
    Y, F = _DummyModel(data=df).run(Y="y", F="cov1")
    assert Y.shape == (5, 1)
    assert F.shape == (5, 2)
