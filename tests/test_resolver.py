from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from cellink import HiData
from cellink._core.dummy_data import sim_adata, sim_gdata
from cellink.at.base_model import BaseModel
from cellink.at.gwas import GWAS
from cellink.at.resolver import get_model_matrix
from cellink.at.structlmm import StructLMM


@pytest.fixture
def hd():
    data = HiData(G=sim_gdata(), C=sim_adata(), group_id="donor_id")
    rng = np.random.default_rng(0)
    data.G.obs["age"] = rng.integers(20, 60, data.G.n_obs).astype(float)
    data.G.obs["phenotype"] = rng.standard_normal(data.G.n_obs)
    return data


def test_get_model_matrix_group_only_formula(hd):
    """A formula referencing only group-level variables must not crash on an empty cell-level dict."""
    y = get_model_matrix(hd, "phenotype - 1", target_level="group")
    assert y.shape == (hd.G.n_obs, 1)
    np.testing.assert_allclose(y.to_numpy().ravel(), hd.G.obs["phenotype"].to_numpy())


def test_get_model_matrix_group_table():
    donors = pd.DataFrame({"age": np.arange(10.0)}, index=pd.Index([f"D{i}" for i in range(10)], name="donor_id"))
    hd = HiData(G=donors, C=sim_adata(), group_id="donor_id")
    X = get_model_matrix(hd, "age", target_level="group")
    assert list(X.index) == list(hd.obs_names)
    np.testing.assert_allclose(X["age"].to_numpy(), donors["age"].to_numpy())
    X = get_model_matrix(hd, "crepeat(age)", target_level="cell")
    np.testing.assert_allclose(X["crepeat(age)"].to_numpy(), hd.C.obs[hd.group_id].map(donors["age"]).to_numpy())


def test_get_model_matrix_cell_only_formula(hd):
    """Symmetric case: a cell-only formula must not crash on an empty group-level dict."""
    x = get_model_matrix(hd, "celltype", target_level="cell")
    assert x.shape[0] == hd.C.n_obs


def test_get_model_matrix_group_aggregation(hd):
    """group_max(<cell-level gene>) aggregates a cell-level variable up to one value per group."""
    X = get_model_matrix(hd, "age + group_max(G0)", target_level="group")
    assert X.shape == (hd.G.n_obs, 3)  # Intercept, age, group_max(G0)
    assert "group_max(G0)" in X.columns
    expected = hd.C[:, "G0"].X.ravel()
    expected = (
        pd.Series(np.asarray(expected).ravel(), index=hd.C.obs_names)
        .groupby(hd.C.obs[hd.group_id], observed=True)
        .max()
    )
    np.testing.assert_allclose(X["group_max(G0)"].to_numpy(), expected.reindex(X.index).to_numpy())


def test_group_aggregation_names_parse_in_formulas(hd):
    """Aggregation names with underscores resolve next to other terms and around nested calls."""
    X = get_model_matrix(hd, "age + group_mean(G0) + group_max(np.square(G1)) - 1", target_level="group")
    assert list(X.columns) == ["age", "group_mean(G0)", "group_max(np.square(G1))"]
    cells = pd.DataFrame(np.asarray(hd.C[:, ["G0", "G1"]].X), index=hd.C.obs_names, columns=["G0", "G1"])
    by_group = cells.groupby(hd.C.obs[hd.group_id], observed=True)
    np.testing.assert_allclose(X["group_mean(G0)"].to_numpy(), by_group["G0"].mean().reindex(X.index).to_numpy())
    expected = np.square(cells["G1"]).groupby(hd.C.obs[hd.group_id], observed=True).max()
    np.testing.assert_allclose(X["group_max(np.square(G1))"].to_numpy(), expected.reindex(X.index).to_numpy())


def test_gwas_resolves_group_aggregations(hd):
    gwas = GWAS(Y="phenotype", F="age + group_mean(G0)", data=hd, target_level="group")
    assert gwas.F.shape == (hd.G.n_obs, 3)  # Intercept, age, group_mean(G0)
    gwas.test_association(np.random.default_rng(1).standard_normal((hd.G.n_obs, 3)))
    assert np.isfinite(gwas.getPv()).all()


def test_get_model_matrix_cell_aggregation_crepeat(hd):
    """crepeat(<group-level var>) broadcasts a group-level variable down to every cell."""
    X = get_model_matrix(hd, "crepeat(age)", target_level="cell")
    assert X.shape == (hd.C.n_obs, 2)  # Intercept, crepeat(age)
    expected = hd.C.obs[hd.group_id].map(hd.G.obs["age"]).to_numpy()
    np.testing.assert_allclose(X["crepeat(age)"].to_numpy(), expected)


def test_get_model_matrix_plain_anndata():
    adata = sim_adata()
    adata.obs["pheno"] = np.random.randn(adata.n_obs)
    y = get_model_matrix(adata, "pheno - 1")
    assert y.shape == (adata.n_obs, 1)
    np.testing.assert_allclose(y.to_numpy().ravel(), adata.obs["pheno"].to_numpy())


def test_get_model_matrix_plain_anndata_rejects_group_agg():
    adata = sim_adata()
    adata.obs["pheno"] = np.random.randn(adata.n_obs)
    with pytest.raises(ValueError, match="not applicable for a single AnnData"):
        get_model_matrix(adata, "group_mean(pheno) - 1")


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


def test_gwas_hidata_formula_runs(hd):
    """GWAS against a HiData with formula strings runs end to end and gives finite p-values."""
    gwas = GWAS(Y="phenotype", F="age", data=hd, target_level="group")
    g = np.random.default_rng(2).standard_normal((hd.G.n_obs, 4))
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
    the HiData branch referenced a nonexistent attribute. Both are
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

    hd = HiData(G=sim_gdata(), C=sim_adata(), group_id="donor_id")
    hd.G.obs["pheno"] = rng.standard_normal(hd.G.n_obs)
    hd.G.obs[snp_cols] = rng.integers(0, 3, size=(hd.G.n_obs, 5)).astype(float)
    pv_hd = skat.run_test(data=hd, Y="pheno", X=snp_formula)
    assert np.isfinite(pv_hd)


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


def test_skat_hidata_defaults_to_group_level():
    """Skat's variants are group-level; a HiData resolves against `.G` without target_level="group"."""
    pytest.importorskip("chiscore", reason="Skat needs chiscore, install with `conda install -c conda-forge chiscore`")
    from cellink.at.skat import Skat

    rng = np.random.default_rng(7)
    snp_cols = [f"snp{i}" for i in range(5)]
    hd = HiData(G=sim_gdata(), C=sim_adata(), group_id="donor_id")
    hd.G.obs["pheno"] = rng.standard_normal(hd.G.n_obs)
    hd.G.obs[snp_cols] = rng.integers(0, 3, size=(hd.G.n_obs, 5)).astype(float)

    skat = Skat(min_threshold=1)
    pv = skat.run_test(data=hd, Y="pheno", X=" + ".join(snp_cols))
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


@pytest.mark.parametrize("agg_func", ["group_mean", "group_max", "group_median", "group_first"])
def test_get_model_matrix_group_aggregation_functions(hd, agg_func):
    X = get_model_matrix(hd, f"{agg_func}(G0) - 1", target_level="group")
    assert X.shape == (hd.G.n_obs, 1)
    assert X.columns[0] == f"{agg_func}(G0)"


def test_get_model_matrix_hidata_requires_target_level(hd):
    with pytest.raises(ValueError, match="target_level"):
        get_model_matrix(hd, "age - 1")


def test_get_model_matrix_rejects_crepeat_in_group_formula(hd):
    with pytest.raises(ValueError, match="crepeat"):
        get_model_matrix(hd, "crepeat(age)", target_level="group")


def test_get_model_matrix_rejects_group_agg_in_cell_formula(hd):
    with pytest.raises(ValueError, match="Group aggregation functions"):
        get_model_matrix(hd, "group_mean(G0)", target_level="cell")


def test_get_model_matrix_dataframe_rejects_group_agg():
    df = pd.DataFrame({"y": np.arange(5.0)})
    with pytest.raises(ValueError, match="not applicable for a pandas DataFrame"):
        get_model_matrix(df, "group_mean(y) - 1")
    with pytest.raises(ValueError, match="not applicable for a pandas DataFrame"):
        get_model_matrix(df, "crepeat(y) - 1")


def test_get_model_matrix_missing_formulaic_raises_helpful_error(monkeypatch):
    _block_module(monkeypatch, "formulaic")
    df = pd.DataFrame({"y": np.arange(5.0)})
    with pytest.raises(ImportError, match=r"cellink-tools\[at\]"):
        get_model_matrix(df, "y - 1")


def _block_module(monkeypatch, name):
    """Simulate `name` being uninstalled, including any of its submodules already cached from earlier tests.

    Blanking only already-cached submodules isn't enough: if `name` itself was never imported by
    an earlier test, it isn't in `sys.modules` yet either, so that lookup finds nothing to patch and
    silently becomes a no-op, letting the real import through. Always blank `name` itself too.
    """
    monkeypatch.setitem(sys.modules, name, None)
    for mod_name in [m for m in sys.modules if m.startswith(f"{name}.")]:
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
