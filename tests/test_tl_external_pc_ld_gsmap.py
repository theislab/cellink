from __future__ import annotations

import gzip

import pandas as pd
import pytest

from cellink._core.donordata import DonorData
from cellink.tl.external._gsmap import load_gsmap_results
from cellink.tl.external._ld import calculate_ld
from cellink.tl.external._pc import calculate_pcs


@pytest.fixture
def dd(adata, gdata):
    gdata.obs["donor_id"] = gdata.obs.index
    return DonorData(G=gdata, C=adata)


def test_calculate_pcs_command_construction(dd, tmp_path):
    prefix = str(tmp_path / "geno")
    cmd = calculate_pcs(dd, prefix, num_pcs=5, run=False)
    assert cmd == f"plink --bfile {prefix} --pca 5 --out {prefix}_pca"


def test_calculate_pcs_custom_out_prefix(dd, tmp_path):
    prefix = str(tmp_path / "geno")
    out = str(tmp_path / "custom_pca_out")
    cmd = calculate_pcs(dd, prefix, out=out, num_pcs=3, run=False)
    assert f"--out {out}" in cmd
    assert f"--pca 3" in cmd


def test_calculate_pcs_save_cmd_file(dd, tmp_path):
    prefix = str(tmp_path / "geno")
    cmd_file = tmp_path / "cmd.sh"
    result = calculate_pcs(dd, prefix, num_pcs=5, run=False, save_cmd_file=str(cmd_file))
    assert result is None
    assert cmd_file.exists()
    assert cmd_file.read_text().strip() == f"plink --bfile {prefix} --pca 5 --out {prefix}_pca"


def test_calculate_pcs_raises_without_plink_binary(dd, tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ImportError, match="plink is required"):
        calculate_pcs(dd, str(tmp_path / "geno"), run=True)


def test_calculate_ld_command_construction(dd, tmp_path):
    prefix = str(tmp_path / "geno")
    cmd = calculate_ld(dd, prefix, window_kb=250, ld_window=500, r2_threshold=0.1, run=False)
    assert cmd == (
        f"plink --bfile {prefix} --r2 --ld-window-kb 250 --ld-window 500 "
        f"--ld-window-r2 0.1 --out {prefix}_ld"
    )


def test_calculate_ld_raises_without_plink_binary_uses_correct_function_name(dd, tmp_path, monkeypatch):
    """Regression test: the error message previously named `calculate_pcs`
    (copy-paste from _pc.py) even when raised from `calculate_ld`."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ImportError, match="calculate_ld"):
        calculate_ld(dd, str(tmp_path / "geno"), run=True)


def test_load_gsmap_results_missing_workdir_returns_none_fields(tmp_path):
    results = load_gsmap_results(tmp_path, sample_name="sample1", trait_name="height")
    assert results["spatial_ldsc"] is None
    assert results["cauchy_combination"] is None
    assert results["report_path"] is None
    assert results["workdir"] == tmp_path


def test_load_gsmap_results_loads_real_files(tmp_path):
    sample_dir = tmp_path / "sample1"
    ldsc_dir = sample_dir / "spatial_ldsc"
    ldsc_dir.mkdir(parents=True)
    ldsc_df = pd.DataFrame({"spot": ["s1", "s2"], "beta": [0.1, 0.2], "se": [0.01, 0.02], "z": [10.0, 10.0], "p": [1e-5, 1e-4]})
    with gzip.open(ldsc_dir / "height_ldsc.csv.gz", "wt") as f:
        ldsc_df.to_csv(f, index=False)

    cauchy_dir = sample_dir / "cauchy_combination"
    cauchy_dir.mkdir(parents=True)
    cauchy_df = pd.DataFrame({"p_cauchy": [0.01, 0.02], "p_median": [0.03, 0.04]}, index=["region1", "region2"])
    cauchy_df.to_csv(cauchy_dir / "height_cauchy.csv")

    report_dir = sample_dir / "report"
    report_dir.mkdir(parents=True)

    results = load_gsmap_results(tmp_path, sample_name="sample1", trait_name="height", annotation="domain")

    assert results["spatial_ldsc"] is not None
    assert list(results["spatial_ldsc"]["spot"]) == ["s1", "s2"]
    assert results["cauchy_combination"] is not None
    assert list(results["cauchy_combination"].index) == ["region1", "region2"]
    assert results["report_path"] == report_dir


def test_load_gsmap_results_falls_back_to_tab_separated(tmp_path):
    """Regression test for the comma/tab auto-detection fallback: gsMap
    sometimes writes tab-separated .gz files despite the .csv.gz name."""
    sample_dir = tmp_path / "sample1"
    ldsc_dir = sample_dir / "spatial_ldsc"
    ldsc_dir.mkdir(parents=True)
    df = pd.DataFrame({"spot": ["s1", "s2"], "z": [1.0, 2.0]})
    with gzip.open(ldsc_dir / "height_ldsc.csv.gz", "wt") as f:
        df.to_csv(f, sep="\t", index=False)

    results = load_gsmap_results(tmp_path, sample_name="sample1", trait_name="height")
    assert results["spatial_ldsc"].shape[1] == 2
    assert list(results["spatial_ldsc"]["spot"]) == ["s1", "s2"]


def test_load_gsmap_results_no_annotation_skips_cauchy(tmp_path):
    sample_dir = tmp_path / "sample1"
    cauchy_dir = sample_dir / "cauchy_combination"
    cauchy_dir.mkdir(parents=True)
    pd.DataFrame({"p_cauchy": [0.01]}).to_csv(cauchy_dir / "height_cauchy.csv")

    results = load_gsmap_results(tmp_path, sample_name="sample1", trait_name="height", annotation=None)
    assert results["cauchy_combination"] is None
