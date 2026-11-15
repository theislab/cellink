from __future__ import annotations

import gzip
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from cellink.tl.external import _ldsc
from cellink.tl.external._ldsc import (
    LDSCRunner,
    compute_ld_scores_with_annotations_from_bimfile,
    compute_ld_scores_with_annotations_from_donor_data,
    estimate_ld_scores_from_bimfile,
    estimate_ld_scores_from_donor_data,
)
from cellink.tl.external._sclinker_utils import compute_ld_scores_for_sclinker


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


def _fake_to_plink(cm_value):
    def fake(gdata, out_prefix, **kwargs):
        pd.DataFrame(
            {
                "chrom": [22, 22],
                "snp": ["22_1_A_G", "22_2_C_T"],
                "cm": [cm_value] * 2,
                "pos": [1, 2],
                "a": "A",
                "b": "G",
            }
        ).to_csv(f"{out_prefix}.bim", sep="\t", header=False, index=False)

    return fake


_DONOR_FUNCS = [
    lambda dd, prefix, runner, **kw: estimate_ld_scores_from_donor_data(dd, prefix, run=False, runner=runner, **kw),
    lambda dd, prefix, runner, **kw: compute_ld_scores_with_annotations_from_donor_data(
        dd, "a.annot.gz", prefix, run=False, runner=runner, **kw
    ),
]


@pytest.mark.parametrize("func", _DONOR_FUNCS)
def test_donor_data_without_genetic_map_uses_1000kb_window(func, runner, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(_ldsc, "to_plink", _fake_to_plink(0.0))
    with caplog.at_level("WARNING"):
        cmd = func(SimpleNamespace(G=None), str(tmp_path / "ld"), runner)["command"]
    assert "--ld-wind-kb 1000" in cmd
    assert "--ld-wind-cm" not in cmd
    assert "cM 0 for every variant" in caplog.text


@pytest.mark.parametrize("func", _DONOR_FUNCS)
def test_donor_data_keeps_explicit_snp_window_and_real_genetic_map(func, runner, tmp_path, monkeypatch):
    monkeypatch.setattr(_ldsc, "to_plink", _fake_to_plink(0.0))
    cmd = func(SimpleNamespace(G=None), str(tmp_path / "ld"), runner, ld_wind_snp=200)["command"]
    assert "--ld-wind-snp 200" in cmd and "--ld-wind-kb" not in cmd

    monkeypatch.setattr(_ldsc, "to_plink", _fake_to_plink(0.5))
    cmd = func(SimpleNamespace(G=None), str(tmp_path / "ld2"), runner)["command"]
    assert "--ld-wind-cm 1.0" in cmd


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


def test_sclinker_ld_scores_keep_reference_panel_without_maf_filter(runner, tmp_path):
    annot_dir = tmp_path / "annotations"
    annot_dir.mkdir()
    with gzip.open(annot_dir / "prog.1.annot.gz", "wt") as f:
        f.write("ANNOT\n0.0\n")

    commands = []
    runner.run_command = lambda cmd, file_paths=None, check=True: commands.append(cmd)
    compute_ld_scores_for_sclinker(
        annotation_prefixes={"prog": {"100kb": str(annot_dir / "prog.")}},
        bim_prefix=str(tmp_path / "1000G.EUR.QC."),
        ld_scores_dir=tmp_path / "ldscores",
        chromosomes=[1],
        n_jobs=1,
        runner=runner,
    )
    assert len(commands) == 1
    assert "--maf" not in commands[0]
    assert "--ld-wind-cm 1.0" in commands[0]


def test_tilde_paths_are_expanded(tmp_path, monkeypatch):
    import subprocess

    from cellink.tl.external import _ldsc2magma

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    _ldsc._ensure_out_prefix_dir("~/ld/out")
    assert (home / "ld").is_dir()

    seen = []
    monkeypatch.setattr(
        _ldsc2magma.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", "")
    )
    _ldsc2magma.run_magma_annotate(
        snp_loc="~/a.loc", gene_loc="~/g.loc", out_prefix="~/magma/x", magma_bin="~/bin/magma"
    )
    assert seen[0][0] == str(home / "bin" / "magma")
    assert str(home / "a.loc") in seen[0] and str(home / "magma" / "x") in seen[0]
    assert (home / "magma").is_dir()
    assert not (tmp_path / "~").exists()
