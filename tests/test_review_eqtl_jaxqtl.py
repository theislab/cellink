import shlex
import subprocess

import numpy as np
import pandas as pd
import pytest

import cellink.tl.external._jaxqtl as jx
from cellink._core.donordata import DonorData
from cellink._core.dummy_data import sim_adata, sim_gdata
from cellink.tl.external._jaxqtl import run_jaxqtl


@pytest.fixture
def dd():
    adata = sim_adata(strategy="poisson")
    gdata = sim_gdata(adata=adata)
    gdata.obs["donor_id"] = gdata.obs.index
    gdata.obs["sex"] = np.resize([1, 2], gdata.n_obs)
    gdata.obs["age"] = np.linspace(30.5, 70.5, gdata.n_obs)
    return DonorData(G=gdata, C=adata)


def _phenotype(prefix):
    df = pd.read_csv(f"{prefix}_phenotype.bed.gz", sep="\t", index_col=3)
    return df.drop(columns=["#chr", "start", "end"]).T


def _per_donor(dd, func, values=None):
    values = dd.C.X if values is None else values
    return (
        pd.DataFrame(values, index=dd.C.obs_names, columns=dd.C.var_names)
        .groupby(dd.C.obs["donor_id"].to_numpy())
        .agg(func)
    )


@pytest.mark.parametrize("model", ["nb", "poisson"])
def test_count_models_get_summed_counts_and_libsize_offset(tmp_path, monkeypatch, dd, model, caplog):
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("WARNING", logger="cellink.tl.external._jaxqtl"):
        cmd = run_jaxqtl(dd, prefix="jx", model=model, expression_pcs=None, run=False)
    assert "--set-offset-from-libsize" in shlex.split(cmd)
    expected = _per_donor(dd, "sum").loc[dd.G.obs_names]
    np.testing.assert_allclose(_phenotype("jx").loc[dd.G.obs_names, dd.C.var_names], expected)
    assert not any("expects raw counts" in m for m in caplog.messages)


def test_gaussian_keeps_mean_without_offset(tmp_path, monkeypatch, dd):
    monkeypatch.chdir(tmp_path)
    cmd = run_jaxqtl(dd, prefix="jx", model="gaussian", expression_pcs=None, run=False)
    assert "--set-offset-from-libsize" not in shlex.split(cmd)
    expected = _per_donor(dd, "mean").loc[dd.G.obs_names]
    np.testing.assert_allclose(_phenotype("jx").loc[dd.G.obs_names, dd.C.var_names], expected, rtol=1e-5)


def test_offset_options(tmp_path, monkeypatch, dd):
    monkeypatch.chdir(tmp_path)
    args = shlex.split(run_jaxqtl(dd, prefix="jx", offset="offsets.tsv", expression_pcs=None, run=False))
    assert args[args.index("--offset") + 1] == "offsets.tsv" and "--set-offset-from-libsize" not in args
    args = shlex.split(run_jaxqtl(dd, prefix="jx", set_offset_from_libsize=False, expression_pcs=None, run=False))
    assert "--set-offset-from-libsize" not in args
    dd.G.obs["log_libsize"] = np.log(_per_donor(dd, "sum").sum(axis=1).loc[dd.G.obs_names].to_numpy())
    kwargs = {"additional_covariates": ["log_libsize"], "offset_name_from_covar": "log_libsize"}
    args = shlex.split(run_jaxqtl(dd, prefix="jx", expression_pcs=None, run=False, **kwargs))
    assert args[args.index("--offset-name-from-covar") + 1] == "log_libsize"
    assert "--set-offset-from-libsize" not in args
    with pytest.raises(ValueError, match="only one of `offset`"):
        run_jaxqtl(dd, prefix="jx", offset="offsets.tsv", set_offset_from_libsize=True, run=False)


def test_counts_layer_and_warning_on_normalized_data(tmp_path, monkeypatch, dd, caplog):
    monkeypatch.chdir(tmp_path)
    counts = dd.C.X.copy()
    dd.C.layers["counts"] = counts
    dd.C.X = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)

    with caplog.at_level("WARNING", logger="cellink.tl.external._jaxqtl"):
        run_jaxqtl(dd, prefix="jx", counts_layer="counts", expression_pcs=None, run=False)
    assert not any("expects raw counts" in m for m in caplog.messages)
    expected = _per_donor(dd, "sum", counts).loc[dd.G.obs_names]
    np.testing.assert_allclose(_phenotype("jx").loc[dd.G.obs_names, dd.C.var_names], expected)

    with caplog.at_level("WARNING", logger="cellink.tl.external._jaxqtl"):
        run_jaxqtl(dd, prefix="jx", expression_pcs=None, run=False)
    assert any("expects raw counts" in m and "dd.C.X" in m for m in caplog.messages)


def test_covariates_keep_float_age_and_reject_missing_sex(tmp_path, monkeypatch, dd):
    monkeypatch.chdir(tmp_path)
    run_jaxqtl(dd, prefix="jx", expression_pcs=None, run=False)
    cov = pd.read_csv("jx_donor_features.tsv", sep="\t", index_col="iid")
    np.testing.assert_allclose(cov["age"], dd.G.obs["age"], rtol=1e-6)
    assert set(cov["sex"]) == {0, 1}

    run_jaxqtl(dd, prefix="jx", encode_sex=False, encode_age=False, expression_pcs=None, run=False)
    assert pd.read_csv("jx_donor_features.tsv", sep="\t").columns.tolist() == ["iid"]

    dd.G.obs["sex"] = dd.G.obs["sex"].astype(float)
    dd.G.obs.iloc[0, dd.G.obs.columns.get_loc("sex")] = np.nan
    with pytest.raises(ValueError, match="have no sex"):
        run_jaxqtl(dd, prefix="jx", expression_pcs=None, run=False)


def test_results_come_from_out_prefix(tmp_path, monkeypatch, dd, pickled_parquet):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jx.shutil, "which", lambda name: f"/usr/bin/{name}")
    commands = []

    def fake_run(cmd, check=False, **kwargs):
        commands.append(cmd)
        out = cmd[cmd.index("--out") + 1]
        pd.DataFrame({"phenotype_id": ["G0"], "pvalue": [0.01]}).to_pickle(f"{out}.cis.score.perm.parquet.gz")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    paths = run_jaxqtl(dd, prefix="jx_in", out="jx_out", expression_pcs=None, read_results=False)
    assert paths == ["jx_out.cis.score.perm.parquet.gz"]
    results = run_jaxqtl(dd, prefix="jx_in", out="jx_out", expression_pcs=None)
    assert results["phenotype_id"].tolist() == ["G0"]
    assert all(isinstance(c, list) for c in commands)
