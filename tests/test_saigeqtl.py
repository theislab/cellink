import shlex

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellink._core import DonorData
from cellink.tl.external._saigeqtl import SAIGEQTLRunner, make_group_file, run_saigeqtl

LOCAL = {
    "execution_mode": "local",
    "rscript_path": "Rscript",
    "step1_script": "step1.R",
    "step2_script": "step2.R",
    "step3_script": "step3.R",
    "makegroup_script": "makeGroupFile.R",
}


@pytest.fixture
def runner():
    return SAIGEQTLRunner(config_dict=LOCAL)


@pytest.fixture
def dd(adata, gdata):
    gdata.obs["donor_id"] = gdata.obs.index
    gdata.obs["sex"] = np.resize([1, 2], gdata.n_obs)
    gdata.obs["age"] = np.linspace(30, 70, gdata.n_obs)
    return DonorData(D=gdata, C=adata)


@pytest.mark.parametrize("sparse", [False, True])
def test_cis_mode_writes_only_the_tested_gene_window(tmp_path, monkeypatch, dd, runner, sparse):
    monkeypatch.chdir(tmp_path)
    if sparse:
        dd.C.X = sp.csr_matrix(dd.C.X)
    gene = dd.C.var_names[0]  # starts at 0, so the window start must be clipped
    commands = run_saigeqtl(dd, gene_col=gene, prefix="sq", window=500_000, run=False, runner=runner)
    ranges = pd.read_csv("sq_ranges.txt", sep="\t", header=None)
    assert ranges.shape[0] == 1
    assert ranges.iloc[0, 1] == 0 and ranges.iloc[0, 2] == int(dd.C.var.loc[gene, "end"]) + 500_000
    assert "--rangestoIncludeFile=sq_ranges.txt" in commands["step2"]
    pheno = pd.read_csv("sq_phenotype.txt", sep="\t")
    np.testing.assert_allclose(pheno[gene], np.asarray(sp.csr_matrix(dd.C.X)[:, 0].todense()).ravel(), rtol=1e-5)


def test_set_based_without_group_file_raises(tmp_path, monkeypatch, dd, runner):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="needs `group_file`"):
        run_saigeqtl(dd, gene_col=dd.C.var_names[0], prefix="sq", analysis_type="set_based", run=False, runner=runner)


def test_missing_sex_raises(tmp_path, monkeypatch, dd, runner):
    monkeypatch.chdir(tmp_path)
    dd.D.obs["sex"] = dd.D.obs["sex"].astype(float)
    dd.D.obs.iloc[0, dd.D.obs.columns.get_loc("sex")] = np.nan
    with pytest.raises(ValueError, match="have no sex"):
        run_saigeqtl(dd, gene_col=dd.C.var_names[0], prefix="sq", run=False, runner=runner)


def test_make_group_file_without_dd_raises(runner):
    with pytest.raises(ValueError, match="needs `dd`"):
        make_group_file(None, region_file="regions.txt", run=False, runner=runner)


def test_arguments_survive_shlex_split(tmp_path, monkeypatch, dd, runner):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "my dir").mkdir()
    prefix = "my dir/sq"
    commands = run_saigeqtl(
        dd,
        gene_col=dd.C.var_names[0],
        prefix=prefix,
        steps=[1, 2, 3],
        weights_beta="1;30",
        group_file="groups; touch pwned",
        analysis_type="set_based",
        run=False,
        runner=runner,
    )
    step1, step2 = shlex.split(commands["step1"]), shlex.split(commands["step2"])
    assert f"--phenoFile={prefix}_phenotype.txt" in step1
    assert f"--bedFile={prefix}.bed" in step2
    assert "--weights.beta=1;30" in step2 and "--groupFile=groups; touch pwned" in step2
    assert f"--assocFile={prefix}_results.txt" in shlex.split(commands["step3"])
