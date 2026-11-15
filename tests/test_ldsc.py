from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from cellink.tl.external import _ldsc
from cellink.tl.external._ldsc import (
    LDSCRunner,
    compute_ld_scores_with_annotations_from_bimfile,
    estimate_ld_scores_from_bimfile,
    estimate_ld_scores_from_donor_data,
)


@pytest.fixture
def runner():
    return LDSCRunner(
        config_dict={
            "execution_mode": "local",
            "ldsc_command": "ldsc.py",
            "make_annot_command": "make_annot.py",
            "munge_command": "munge_sumstats.py",
        }
    )


def _plain(runner, tmp_path, **kwargs):
    return estimate_ld_scores_from_bimfile("ref/chr22", str(tmp_path / "out"), run=False, runner=runner, **kwargs)[
        "command"
    ]


def _annot(runner, tmp_path, **kwargs):
    return compute_ld_scores_with_annotations_from_bimfile(
        "ref/chr22", "a.annot.gz", str(tmp_path / "out"), run=False, runner=runner, **kwargs
    )["command"]


@pytest.mark.parametrize("build", [_plain, _annot])
def test_default_window_is_1cm_and_no_maf_filter(build, runner, tmp_path):
    cmd = build(runner, tmp_path)
    assert "--ld-wind-cm 1.0" in cmd
    assert "--maf" not in cmd  # ldsc's default, as for the baseline-LD scores


@pytest.mark.parametrize("build", [_plain, _annot])
def test_any_maf_min_is_passed(build, runner, tmp_path):
    assert "--maf 0.01" in build(runner, tmp_path, maf_min=0.01)


@pytest.mark.parametrize("build", [_plain, _annot])
@pytest.mark.parametrize(("kwarg", "flag"), [("ld_wind_kb", "--ld-wind-kb 500"), ("ld_wind_snp", "--ld-wind-snp 500")])
def test_kb_or_snp_window_alone_is_accepted(build, kwarg, flag, runner, tmp_path):
    cmd = build(runner, tmp_path, **{kwarg: 500})
    assert flag in cmd
    assert "--ld-wind-cm" not in cmd


@pytest.mark.parametrize("build", [_plain, _annot])
def test_two_windows_still_raise(build, runner, tmp_path):
    with pytest.raises(ValueError, match="Only one of"):
        build(runner, tmp_path, ld_wind_cm=1.0, ld_wind_kb=500)


@pytest.mark.parametrize("build", [_plain, _annot])
def test_maf_none_passes_no_filter_and_other_values_pass_through(build, runner, tmp_path):
    assert "--maf" not in build(runner, tmp_path, maf_min=None)
    assert "--maf 0.05" in build(runner, tmp_path, maf_min=0.05)


def test_donor_data_with_real_plink_export_is_not_whole_chromosome(runner, tmp_path):
    rng = np.random.default_rng(0)
    var = pd.DataFrame(
        {"chrom": ["22"] * 5, "pos": np.arange(1, 6) * 1000, "a0": ["A"] * 5, "a1": ["G"] * 5},
        index=[f"22_{i * 1000}_A_G" for i in range(1, 6)],
    )
    gdata = AnnData(
        X=rng.integers(0, 3, size=(4, 5)).astype(np.float32),
        obs=pd.DataFrame(index=[f"d{i}" for i in range(4)]),
        var=var,
    )
    cmd = estimate_ld_scores_from_donor_data(SimpleNamespace(G=gdata), str(tmp_path / "ld"), run=False, runner=runner)[
        "command"
    ]
    assert "--ld-wind-kb 1000" in cmd


def test_tilde_paths_are_expanded(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    _ldsc._ensure_out_prefix_dir("~/ld/out")
    assert (home / "ld").is_dir()
    assert not (tmp_path / "~").exists()
