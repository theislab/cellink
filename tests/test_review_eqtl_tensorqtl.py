import importlib.machinery
import importlib.util
import os
import shlex
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

import cellink.tl.external._tensorqtl as tq
from cellink._core.donordata import DonorData
from cellink.tl.external._tensorqtl import _check_qvalue_stack, run_tensorqtl


@pytest.fixture
def dd(adata, gdata):
    gdata.obs["donor_id"] = gdata.obs.index
    gdata.obs["sex"] = np.resize([1, 2], gdata.n_obs)
    gdata.obs["age"] = np.linspace(30, 70, gdata.n_obs)
    return DonorData(G=gdata, C=adata)


def _read_covariates(prefix):
    return pd.read_csv(f"{prefix}_donor_features.tsv", sep="\t", index_col=0).T


def test_missing_sex_raises_instead_of_code_minus_one(tmp_path, dd):
    dd.G.obs["sex"] = dd.G.obs["sex"].astype(float)
    dd.G.obs.iloc[0, dd.G.obs.columns.get_loc("sex")] = np.nan
    with pytest.raises(ValueError, match="1 donor\\(s\\) have no sex"):
        run_tensorqtl(dd, mode="trans", prefix=str(tmp_path / "tq"), run=False)


def test_missing_covariate_values_raise(tmp_path, dd):
    dd.G.obs.iloc[1, dd.G.obs.columns.get_loc("age")] = np.nan
    with pytest.raises(ValueError, match="age"):
        run_tensorqtl(dd, mode="trans", prefix=str(tmp_path / "tq"), run=False)


def test_run_false_writes_pgen_and_builds_command(tmp_path, dd, monkeypatch):
    pytest.importorskip("pgenlib")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    prefix = str(tmp_path / "my dir" / "tq")
    os.makedirs(os.path.dirname(prefix))
    save = tmp_path / "job.sh"
    cmd = run_tensorqtl(dd, mode="trans", prefix=prefix, run=False, save_cmd_file=str(save))
    assert calls == []
    assert all(os.path.isfile(f"{prefix}.{ext}") for ext in ("pgen", "pvar", "psam"))
    args = shlex.split(cmd)
    assert args[:3] == [sys.executable, "-m", "cellink.tl.external._tensorqtl_main"]
    assert args[3:6] == [prefix, f"{prefix}_phenotype.bed.gz", prefix]
    assert save.read_text().strip() == cmd


@pytest.fixture
def fake_cli(monkeypatch):
    """Pretend tensorqtl is installed; record its commands and write the outputs it would."""
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(tq.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: importlib.machinery.ModuleSpec(name, None)
        if name == "tensorqtl"
        else real_find_spec(name, *a),
    )
    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(cmd)
        assert isinstance(cmd, list)  # no shell
        assert all(os.path.isfile(f"{cmd[3]}.{ext}") for ext in ("pgen", "pvar", "psam"))
        prefix = cmd[5]
        pd.DataFrame({"phenotype_id": ["G0"], "pval": [0.1]}).to_pickle(f"{prefix}.trans_qtl_pairs.parquet")
        open(f"{prefix}.tensorQTL.trans.log", "w").close()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_cli_runs_tensorqtl_with_current_interpreter_and_returns_paths(tmp_path, dd, fake_cli):
    prefix = str(tmp_path / "tq")
    pytest.importorskip("pgenlib")
    paths = run_tensorqtl(dd, mode="trans", prefix=prefix, read_results=False)
    (tensorqtl,) = fake_cli
    assert tensorqtl[3] == prefix
    assert not any(os.path.exists(f"{prefix}.{ext}") for ext in ("pgen", "pvar", "psam"))
    assert tensorqtl[:3] == [sys.executable, "-m", "cellink.tl.external._tensorqtl_main"]
    assert paths == [f"{prefix}.tensorQTL.trans.log", f"{prefix}.trans_qtl_pairs.parquet"]


def test_cli_cis_raises_before_running_without_qvalue_stack(tmp_path, dd, fake_cli, monkeypatch):
    monkeypatch.setattr(tq.shutil, "which", lambda name: None if name == "R" else f"/usr/bin/{name}")
    with pytest.raises(RuntimeError, match="R is not on PATH"):
        run_tensorqtl(dd, mode="cis", prefix=str(tmp_path / "tq"))
    assert fake_cli == []


def test_check_qvalue_stack_reports_broken_rpy2(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(tq.shutil, "which", lambda name: "/usr/bin/R")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: importlib.machinery.ModuleSpec(name, None) if name == "rpy2" else real_find_spec(name, *a),
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "Error: symbol not found\n")
    )
    with pytest.raises(RuntimeError, match="symbol not found"):
        _check_qvalue_stack()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "", ""))
    _check_qvalue_stack()


@pytest.fixture
def fake_tensorqtl(monkeypatch):
    """A stand-in tensorqtl package recording the keyword arguments of the mapping functions."""
    calls = {}

    def recorder(name, result):
        def fn(*args, **kwargs):
            calls[name] = kwargs
            return result() if callable(result) else result

        return fn

    pkg = types.ModuleType("tensorqtl")
    pkg.__spec__ = importlib.machinery.ModuleSpec("tensorqtl", None)
    pkg.cis = types.SimpleNamespace(
        map_cis=recorder("map_cis", lambda: pd.DataFrame({"pval_beta": [0.1]}, index=["G0"])),
        map_nominal=recorder("map_nominal", None),
        map_independent=recorder("map_independent", pd.DataFrame()),
    )
    pkg.post = types.SimpleNamespace(calculate_qvalues=lambda res, **k: res.__setitem__("qval", 0.5))
    pkg.trans = types.SimpleNamespace(
        map_trans=recorder("map_trans", pd.DataFrame()), filter_cis=lambda df, *a, **k: df
    )
    pkg.susie = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "tensorqtl", pkg)
    monkeypatch.setitem(sys.modules, "tensorqtl.genotypeio", types.ModuleType("tensorqtl.genotypeio"))
    monkeypatch.setattr(tq, "_check_qvalue_stack", lambda: None)
    return calls


def test_python_api_forwards_maf_threshold(tmp_path, dd, fake_tensorqtl, pickled_parquet):
    results = run_tensorqtl(dd, mode="cis", maf_threshold=0.05, use_python_api=True)
    assert fake_tensorqtl["map_cis"]["maf_threshold"] == 0.05
    assert "qval" in results

    run_tensorqtl(dd, mode="cis_independent", cis_output=results, maf_threshold=0.05, use_python_api=True)
    assert fake_tensorqtl["map_independent"]["maf_threshold"] == 0.05

    prefix = str(tmp_path / "nom")
    pd.DataFrame({"phenotype_id": ["G0"]}).to_pickle(f"{prefix}.cis_qtl_pairs.1.parquet")
    run_tensorqtl(dd, mode="cis_nominal", prefix=prefix, maf_threshold=0.05, use_python_api=True)
    assert fake_tensorqtl["map_nominal"]["maf_threshold"] == 0.05


def test_python_api_cis_checks_qvalue_stack_before_mapping(dd, fake_tensorqtl, monkeypatch):
    def fail():
        raise RuntimeError("no qvalue")

    monkeypatch.setattr(tq, "_check_qvalue_stack", fail)
    with pytest.raises(RuntimeError, match="no qvalue"):
        run_tensorqtl(dd, mode="cis", use_python_api=True)
    assert "map_cis" not in fake_tensorqtl
