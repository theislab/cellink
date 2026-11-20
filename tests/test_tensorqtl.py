import os
import shlex
import sys

import numpy as np
import pandas as pd
import pytest

from cellink._core import DonorData
from cellink.tl.external._tensorqtl import run_tensorqtl
from cellink.tl.external._tensorqtl_main import read_phenotype_bed


def _write_bed(path, chrom, start, end):
    df = pd.DataFrame(
        {
            "#chr": chrom,
            "start": start,
            "end": end,
            "gene_id": [f"g{i}" for i in range(len(chrom))],
            "d1": 1.0,
            "d2": 2.0,
        }
    )
    df.to_csv(path, sep="\t", index=False)
    return path


def test_read_phenotype_bed_accepts_sorted_bed(tmp_path):
    bed = _write_bed(tmp_path / "p.bed.gz", ["22", "22", "22", "1", "1"], [10, 10, 30, 5, 7], [11, 12, 31, 6, 8])
    phenotype_df, pos_df = read_phenotype_bed(str(bed))
    assert list(phenotype_df.columns) == ["d1", "d2"]
    assert list(phenotype_df.index) == ["g0", "g1", "g2", "g3", "g4"]
    assert list(pos_df.columns) == ["chr", "start", "end"]
    assert pos_df["chr"].tolist() == ["22", "22", "22", "1", "1"]
    assert pos_df["start"].tolist() == [11, 11, 31, 6, 8]  # 1-based


def test_read_phenotype_bed_single_position_column(tmp_path):
    bed = _write_bed(tmp_path / "p.bed", ["1", "1"], [4, 9], [5, 10])
    _, pos_df = read_phenotype_bed(str(bed))
    assert list(pos_df.columns) == ["chr", "pos"] and pos_df["pos"].tolist() == [5, 10]


@pytest.mark.parametrize(
    "chrom, start",
    [(["1", "1"], [20, 10]), (["1", "2", "1"], [10, 10, 20])],
    ids=["unsorted-positions", "split-chromosome"],
)
def test_read_phenotype_bed_rejects_unsorted_bed(tmp_path, chrom, start):
    bed = _write_bed(tmp_path / "p.bed", chrom, start, [s + 1 for s in start])
    with pytest.raises(AssertionError, match="must be sorted"):
        read_phenotype_bed(str(bed))


def test_pandas_plink_stub_points_to_pgen():
    from cellink.tl.external._tensorqtl import _pandas_plink_read_plink

    with pytest.raises(ImportError, match="PGEN"):
        _pandas_plink_read_plink("genotypes")


def test_run_tensorqtl_writes_the_genotypes_as_pgen(tmp_path, adata, gdata):
    import shlex

    import numpy as np

    from cellink._core import DonorData
    from cellink.tl.external._tensorqtl import run_tensorqtl

    pgenlib = pytest.importorskip("pgenlib")
    gdata.obs["donor_id"] = gdata.obs.index
    gdata.obs["sex"] = np.resize([1, 2], gdata.n_obs)
    gdata.obs["age"] = np.linspace(30, 70, gdata.n_obs)
    X = np.asarray(gdata.X, dtype=np.float32)
    X[0, 1] = np.nan
    gdata.X = X
    dd = DonorData(D=gdata, C=adata)
    prefix = str(tmp_path / "tq")
    cmd = run_tensorqtl(dd, mode="trans", prefix=prefix, run=False)
    assert shlex.split(cmd)[3] == prefix

    reader = pgenlib.PgenReader(f"{prefix}.pgen".encode())
    n_variants = reader.get_variant_ct()
    calls_out = np.empty((n_variants, reader.get_raw_sample_ct()), dtype=np.int8)
    reader.read_range(0, n_variants, calls_out)
    psam = pd.read_csv(f"{prefix}.psam", sep="\t", dtype=str)
    G = dd.D[psam["IID" if "IID" in psam.columns else "#IID"].to_numpy()]
    expected = np.asarray(G.X, dtype=np.float32)
    np.testing.assert_array_equal(calls_out.T, np.where(np.isnan(expected), -9, expected))
    pvar = pd.read_csv(f"{prefix}.pvar", sep="\t", dtype=str)
    assert list(pvar["REF"]) == list(G.var["a0"].astype(str))
    assert list(pvar["ALT"]) == list(G.var["a1"].astype(str))


@pytest.fixture
def dd(adata, gdata):
    gdata.obs["donor_id"] = gdata.obs.index
    gdata.obs["sex"] = np.resize([1, 2], gdata.n_obs)
    gdata.obs["age"] = np.linspace(30, 70, gdata.n_obs)
    return DonorData(D=gdata, C=adata)


def _read_covariates(prefix):
    return pd.read_csv(f"{prefix}_donor_features.tsv", sep="\t", index_col=0).T


def test_covariates_have_no_intercept_and_keep_first_donor(tmp_path, dd):
    prefix = str(tmp_path / "tq")
    run_tensorqtl(dd, mode="trans", prefix=prefix, run=False)
    cov = _read_covariates(prefix)
    assert list(cov.columns) == ["sex", "age"]
    assert np.allclose(cov["age"].mean(), 0, atol=1e-5)  # z-scored, not truncated

    # tensorQTL's Residualizer: QR of the centered covariates, then project out
    c = cov.to_numpy(dtype=float)
    q, _ = np.linalg.qr(c - c.mean(0))
    y = np.random.default_rng(0).normal(size=len(c))
    y0 = y - y.mean()
    residual = y0 - q @ (q.T @ y0)
    assert abs(residual[0]) > 1e-6


def test_missing_sex_raises_instead_of_code_minus_one(tmp_path, dd):
    dd.D.obs["sex"] = dd.D.obs["sex"].astype(float)
    dd.D.obs.iloc[0, dd.D.obs.columns.get_loc("sex")] = np.nan
    with pytest.raises(ValueError, match="1 donor\\(s\\) have no sex"):
        run_tensorqtl(dd, mode="trans", prefix=str(tmp_path / "tq"), run=False)


def test_missing_covariate_values_raise(tmp_path, dd):
    dd.D.obs.iloc[1, dd.D.obs.columns.get_loc("age")] = np.nan
    with pytest.raises(ValueError, match="age"):
        run_tensorqtl(dd, mode="trans", prefix=str(tmp_path / "tq"), run=False)


def test_constant_covariate_is_dropped(tmp_path, dd, caplog):
    dd.D.obs["sex"] = 1
    prefix = str(tmp_path / "tq")
    with caplog.at_level("WARNING", logger="cellink.tl.external._tensorqtl"):
        run_tensorqtl(dd, mode="trans", prefix=prefix, run=False)
    assert list(_read_covariates(prefix).columns) == ["age"]
    assert any("constant covariate" in m for m in caplog.messages)


def test_without_covariates_no_covariates_file_is_passed(tmp_path, dd):
    prefix = str(tmp_path / "tq")
    cmd = run_tensorqtl(dd, mode="trans", prefix=prefix, encode_sex=False, encode_age=False, run=False)
    assert "--covariates" not in shlex.split(cmd)
    assert not os.path.exists(f"{prefix}_donor_features.tsv")
    with pytest.raises(ValueError, match="needs at least one covariate"):
        run_tensorqtl(
            dd, mode="cis_susie", cis_output="x.txt.gz", prefix=prefix, encode_sex=False, encode_age=False, run=False
        )


def test_run_false_writes_pgen_and_builds_command(tmp_path, dd):
    pytest.importorskip("pgenlib")
    prefix = str(tmp_path / "my dir" / "tq")
    os.makedirs(os.path.dirname(prefix))
    save = tmp_path / "job.sh"
    cmd = run_tensorqtl(dd, mode="trans", prefix=prefix, run=False, save_cmd_file=str(save))
    assert all(os.path.isfile(f"{prefix}.{ext}") for ext in ("pgen", "pvar", "psam"))
    args = shlex.split(cmd)
    assert args[:3] == [sys.executable, "-m", "cellink.tl.external._tensorqtl_main"]
    assert args[3:6] == [prefix, f"{prefix}_phenotype.bed.gz", prefix]
    assert save.read_text().strip() == cmd
