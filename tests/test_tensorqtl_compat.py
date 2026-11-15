import pandas as pd
import pytest

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


def test_run_tensorqtl_writes_the_genotypes_as_pgen(tmp_path, adata, gdata, monkeypatch):
    import shlex
    import subprocess

    import numpy as np

    from cellink._core.donordata import DonorData
    from cellink.tl.external._tensorqtl import run_tensorqtl

    pgenlib = pytest.importorskip("pgenlib")
    gdata.obs["donor_id"] = gdata.obs.index
    gdata.obs["sex"] = np.resize([1, 2], gdata.n_obs)
    gdata.obs["age"] = np.linspace(30, 70, gdata.n_obs)
    X = np.asarray(gdata.X, dtype=np.float32)
    X[0, 1] = np.nan
    gdata.X = X
    dd = DonorData(G=gdata, C=adata)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    prefix = str(tmp_path / "tq")
    cmd = run_tensorqtl(dd, mode="trans", prefix=prefix, run=False)
    assert calls == []  # no plink2 conversion
    assert shlex.split(cmd)[3] == prefix

    reader = pgenlib.PgenReader(f"{prefix}.pgen".encode())
    n_variants = reader.get_variant_ct()
    calls_out = np.empty((n_variants, reader.get_raw_sample_ct()), dtype=np.int8)
    reader.read_range(0, n_variants, calls_out)
    psam = pd.read_csv(f"{prefix}.psam", sep="\t", dtype=str)
    G = dd.G[psam["IID" if "IID" in psam.columns else "#IID"].to_numpy()]
    expected = np.asarray(G.X, dtype=np.float32)
    np.testing.assert_array_equal(calls_out.T, np.where(np.isnan(expected), -9, expected))
    pvar = pd.read_csv(f"{prefix}.pvar", sep="\t", dtype=str)
    assert list(pvar["REF"]) == list(G.var["a0"].astype(str))
    assert list(pvar["ALT"]) == list(G.var["a1"].astype(str))
