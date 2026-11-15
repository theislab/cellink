from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellink.io import read_plink, to_plink

DATA = Path("tests/data")


def _gdata(X, sex=None):
    n_obs, n_vars = X.shape
    rng = np.random.default_rng(1)
    bases = np.array(list("ACGT"))
    a0 = rng.choice(bases, n_vars)
    a1 = np.array([rng.choice(bases[bases != r]) for r in a0])
    pos = 1000 + np.arange(n_vars) * 7
    var = pd.DataFrame({"chrom": ["2"] * n_vars, "pos": pos, "a0": a0, "a1": a1})
    var.index = [f"2_{p}_{r}_{a}" for p, r, a in zip(pos, a0, a1, strict=True)]
    obs = pd.DataFrame({"donor_id": [f"d{i}" for i in range(n_obs)]})
    obs.index = obs["donor_id"].to_numpy()
    if sex is not None:
        obs["sex"] = sex
    return ad.AnnData(X=X, obs=obs, var=var)


def _random_X(n_obs=9, n_vars=13, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n_obs, n_vars)).astype("float32")
    X[rng.random(X.shape) < 0.1] = np.nan
    return X


def _decode_bed_by_spec(bed: bytes, n_samples: int, n_variants: int) -> np.ndarray:
    """Copies of the .bim column-5 allele (A1) per the PLINK 1 spec, independent of cellink.

    00 = homozygous A1, 01 = missing, 10 = heterozygous, 11 = homozygous A2.
    """
    assert bed[:3] == b"\x6c\x1b\x01"
    per_variant = (n_samples + 3) // 4
    a1_copies = {0b00: 2.0, 0b01: np.nan, 0b10: 1.0, 0b11: 0.0}
    out = np.empty((n_samples, n_variants))
    for v in range(n_variants):
        block = bed[3 + v * per_variant : 3 + (v + 1) * per_variant]
        for s in range(n_samples):
            out[s, v] = a1_copies[(block[s // 4] >> (2 * (s % 4))) & 0b11]
    return out


def test_to_plink_writes_alt_as_a1_and_bed_counts_it(tmp_path):
    """plink2 --bfile, tensorQTL and jaxqtl count the column-5 allele (A1) as ALT."""
    X = _random_X()
    gdata = _gdata(X)
    to_plink(gdata, output_prefix=tmp_path / "g")

    bim = pd.read_csv(tmp_path / "g.bim", sep="\t", header=None, dtype=str)
    assert list(bim[4]) == list(gdata.var["a1"])  # column 5 = A1 = ALT
    assert list(bim[5]) == list(gdata.var["a0"])  # column 6 = A2 = REF
    a1_copies = _decode_bed_by_spec((tmp_path / "g.bed").read_bytes(), *X.shape)
    np.testing.assert_array_equal(a1_copies, X)


def test_read_plink_roundtrip_keeps_x_alleles_and_index(tmp_path):
    pytest.importorskip("bed_reader")
    X = _random_X(n_obs=6, n_vars=21, seed=3)
    gdata = _gdata(X)
    to_plink(gdata, output_prefix=tmp_path / "rt")
    back = read_plink(tmp_path / "rt")
    np.testing.assert_array_equal(np.asarray(back.X.compute()), X)
    assert list(back.var.index) == list(gdata.var.index)
    assert list(back.var["a0"]) == list(gdata.var["a0"])
    assert list(back.var["a1"]) == list(gdata.var["a1"])
    assert list(back.obs.index) == list(gdata.obs.index)


def test_read_plink_follows_plink2_orientation():
    """tests/data/simulated_genotype_calls.{bed,bim,fam} and .{pgen,pvar,psam} hold the same
    calls; plink2 put ALT in .bim column 5. read_plink must return the pgen ALT counts and
    the pvar REF/ALT."""
    pytest.importorskip("bed_reader")
    pgenlib = pytest.importorskip("pgenlib")
    gdata = read_plink(DATA / "simulated_genotype_calls")
    reader = pgenlib.PgenReader(bytes(str(DATA / "simulated_genotype_calls.pgen"), "utf-8"))
    alt_counts = np.zeros((100, 1000), dtype=np.int8)
    reader.read_range(0, 1000, geno_int_out=alt_counts, sample_maj=1)
    np.testing.assert_array_equal(np.asarray(gdata.X.compute()), alt_counts)

    pvar = pd.read_csv(DATA / "simulated_genotype_calls.pvar", sep="\t", dtype=str)
    assert list(gdata.var["a0"]) == list(pvar["REF"])
    assert list(gdata.var["a1"]) == list(pvar["ALT"])


def test_read_plink_count_a1_false_counts_column_6(tmp_path):
    """count_a1=False counts .bim column 6, for files with REF in column 5."""
    pytest.importorskip("bed_reader")
    X = _random_X(seed=5)
    gdata = _gdata(X)
    to_plink(gdata, output_prefix=tmp_path / "g")
    col6 = read_plink(tmp_path / "g", count_a1=False)
    np.testing.assert_array_equal(np.asarray(col6.X.compute()), 2 - X)
    assert list(col6.var["a0"]) == list(gdata.var["a1"])
    assert list(col6.var["a1"]) == list(gdata.var["a0"])


@pytest.mark.parametrize(
    ("sex", "sex_map", "expected"),
    [
        ([1, 0, 0, 1], None, [1, 2, 2, 1]),  # get_onek1k coding: 1 = male, 0 = female
        ([1, 2, 0, 2], None, [1, 2, 0, 2]),  # PLINK codes
        ([1.0, np.nan, 0.0, 1.0], None, [1, 0, 2, 1]),  # missing -> unknown
        (["male", "Female", "F", "unknown"], None, [1, 2, 2, 0]),
        (pd.Categorical(["female", "male", "male", "female"]), None, [2, 1, 1, 2]),
        (["M", "F", "M", "?"], {"M": 1, "F": 2}, [1, 2, 1, 0]),
        ([0, 1, 1, 0], {0: 0, 1: 1}, [0, 1, 1, 0]),  # PLINK-coded without females
        ([1, -1, 2, -9], None, [1, 0, 2, 0]),  # -1 from read_plink, -9 in PLINK: unknown
    ],
)
def test_to_plink_sex_codes(tmp_path, sex, sex_map, expected):
    pytest.importorskip("bed_reader")
    gdata = _gdata(_random_X(n_obs=4, n_vars=3), sex=sex)
    to_plink(gdata, output_prefix=tmp_path / "s", donor_sex_map=sex_map)
    fam = pd.read_csv(tmp_path / "s.fam", sep=r"\s+", header=None, dtype=str)
    assert [int(v) for v in fam[4]] == expected
    obs = read_plink(tmp_path / "s").obs
    assert list(obs["sex"]) == [c if c else -1 for c in expected]


def test_to_plink_rejects_uninterpretable_sex(tmp_path):
    gdata = _gdata(_random_X(n_obs=4, n_vars=3), sex=[1, 2, 3, 1])
    with pytest.raises(ValueError, match="donor_sex_map"):
        to_plink(gdata, output_prefix=tmp_path / "s")


def test_to_plink_parent_ids_and_donor_ids_roundtrip(tmp_path):
    pytest.importorskip("bed_reader")
    gdata = _gdata(_random_X(n_obs=3, n_vars=2))
    gdata.obs["donor_paternal_id"] = ["P1", "P2", "P3"]
    gdata.obs["donor_maternal_id"] = ["M1", "M2", "M3"]
    to_plink(gdata, output_prefix=tmp_path / "p")
    obs = read_plink(tmp_path / "p").obs
    assert list(obs.index) == ["d0", "d1", "d2"]
    assert list(obs["paternal_id"]) == ["P1", "P2", "P3"]
    assert list(obs["maternal_id"]) == ["M1", "M2", "M3"]


def test_read_plink_to_plink_roundtrip_with_unknown_sex_and_chromosome(tmp_path):
    pytest.importorskip("bed_reader")
    gdata = read_plink(DATA / "simulated_genotype_calls")
    to_plink(gdata, output_prefix=tmp_path / "rt")
    back = read_plink(tmp_path / "rt")
    np.testing.assert_array_equal(np.asarray(back.X.compute()), np.asarray(gdata.X.compute()))
    assert list(back.var.index) == list(gdata.var.index)
    assert list(back.obs.index) == list(gdata.obs.index)
    assert list(back.obs["sex"]) == list(gdata.obs["sex"]) == [-1] * 100


@pytest.mark.parametrize(
    ("phenotypes", "expected"),
    [
        (["1", "2", "-9", "0"], [1, 2, -1, -1]),
        (["0.5", "-9", "NA", "2"], [0.5, np.nan, np.nan, 2]),
        (["-9", "0", "1", "3"], [np.nan, 0, 1, 3]),
    ],
)
def test_read_plink_keeps_quantitative_phenotypes(tmp_path, phenotypes, expected):
    pytest.importorskip("bed_reader")
    to_plink(_gdata(_random_X(n_obs=4)), output_prefix=tmp_path / "p")
    fam = (tmp_path / "p.fam").read_text().splitlines()
    rows = [" ".join(line.split()[:5] + [p]) for line, p in zip(fam, phenotypes, strict=True)]
    (tmp_path / "p.fam").write_text("\n".join(rows) + "\n")
    expected = np.asarray(expected, dtype=float)
    obs = read_plink(tmp_path / "p").obs
    np.testing.assert_array_equal(obs["phenotype"].to_numpy(dtype=float), expected)
    files = {f"{ext}_path": tmp_path / f"p.{ext}" for ext in ("bed", "bim", "fam")}
    obs = read_plink(**files, fam_sep=" ").obs
    np.testing.assert_array_equal(obs["phenotype"].to_numpy(dtype=float), expected)
