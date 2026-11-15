from __future__ import annotations

import gzip

import pandas as pd

from cellink.tl.external._gsmap import load_gsmap_results


def _write_ldsc(path, z):
    with gzip.open(path, "wt") as f:
        pd.DataFrame({"spot": ["a", "b"], "z": [z, z]}).to_csv(f, index=False)


def test_load_gsmap_results_ignores_other_traits(tmp_path, caplog):
    ldsc_dir = tmp_path / "s" / "spatial_ldsc"
    ldsc_dir.mkdir(parents=True)
    _write_ldsc(ldsc_dir / "s_Height.csv.gz", 9.0)
    cauchy_dir = tmp_path / "s" / "cauchy_combination"
    cauchy_dir.mkdir()
    pd.DataFrame({"p_cauchy": [0.01]}, index=["r"]).to_csv(cauchy_dir / "s_Height.Cauchy.csv.gz")

    with caplog.at_level("WARNING"):
        res = load_gsmap_results(tmp_path, "s", "IQ", annotation="domain")
    assert res["spatial_ldsc"] is None and res["cauchy_combination"] is None
    assert "No spatial LDSC results for trait 'IQ'" in caplog.text
    assert "No Cauchy combination results for trait 'IQ'" in caplog.text

    _write_ldsc(ldsc_dir / "s_IQ.csv.gz", 2.0)
    assert load_gsmap_results(tmp_path, "s", "IQ")["spatial_ldsc"]["z"].tolist() == [2.0, 2.0]


def test_load_gsmap_results_expands_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    ldsc_dir = tmp_path / "work" / "s" / "spatial_ldsc"
    ldsc_dir.mkdir(parents=True)
    _write_ldsc(ldsc_dir / "s_IQ.csv.gz", 1.0)
    res = load_gsmap_results("~/work", "s", "IQ")
    assert res["workdir"] == tmp_path / "work"
    assert res["spatial_ldsc"] is not None
