import shutil
from pathlib import Path

import numpy as np
import pytest

pgenlib = pytest.importorskip("pgenlib")

from cellink.io import read_pgen_zarr, stream_pgen_to_zarr

DATA = Path("tests/data")

# (variants, samples) ALT counts, -9 = missing (pgenlib's code)
_GENO = np.array([[0, 1, 2, -9, 1], [2, 2, -9, 0, 0], [1, 1, 1, 1, 1]], dtype=np.int8)


def _write_pgen(base: Path, psam: str, geno: np.ndarray = _GENO) -> str:
    n_variants, n_samples = geno.shape
    writer = pgenlib.PgenWriter(bytes(f"{base}.pgen", "utf-8"), n_samples, n_variants, nonref_flags=False)
    for row in geno:
        writer.append_biallelic(np.ascontiguousarray(row))
    writer.close()
    pvar = "#CHROM\tPOS\tID\tREF\tALT\n" + "".join(f"1\t{100 + i}\tv{i}\tA\tG\n" for i in range(n_variants))
    Path(f"{base}.pvar").write_text(pvar)
    Path(f"{base}.psam").write_text(psam)
    return f"{base}.pgen"


_PSAM_IID = "#IID\tSEX\n" + "".join(f"S{i}\tNA\n" for i in range(5))


def test_psam_without_fid_is_indexed_by_iid(tmp_path):
    adata = stream_pgen_to_zarr(
        str(DATA / "simulated_genotype_calls.pgen"), str(tmp_path / "g.zarr"), return_adata=True
    )
    assert list(adata.obs.index[:3]) == ["S0", "S1", "S2"]
    assert list(read_pgen_zarr(tmp_path / "g.zarr").obs.index[:3]) == ["S0", "S1", "S2"]


def test_psam_with_fid_is_indexed_by_fid_and_iid(tmp_path):
    psam = "#FID\tIID\tSEX\n" + "".join(f"F{i}\tS{i}\t1\n" for i in range(5))
    pgen = _write_pgen(tmp_path / "fid", psam)
    adata = stream_pgen_to_zarr(pgen, str(tmp_path / "fid.zarr"), return_adata=True)
    assert list(adata.obs.index) == [f"F{i}_S{i}" for i in range(5)]


def test_two_bit_store_keeps_missing_calls_missing(tmp_path):
    # 2-bit code 3 marks missing calls: -1 with backend="zarr", nan otherwise, like the int8 sentinel
    pgen = _write_pgen(tmp_path / "miss2bit", _PSAM_IID)
    out = tmp_path / "miss2bit.zarr"
    with pytest.warns(UserWarning, match="keeps them missing"):
        stream_pgen_to_zarr(pgen, str(out))
    expected = _GENO.T.astype(np.float32)
    n_missing = int((expected < 0).sum())
    raw = read_pgen_zarr(out, backend="zarr")
    assert int(raw.uns["n_missing"]) == n_missing > 0
    np.testing.assert_array_equal(np.asarray(raw.X[:, :]), np.where(expected < 0, -1, expected))
    X = read_pgen_zarr(out).X
    X = np.asarray(X.compute() if hasattr(X, "compute") else X)
    expected[expected < 0] = np.nan
    np.testing.assert_array_equal(X, expected)


@pytest.mark.parametrize("sparse", [False, True])
def test_missing_calls_stay_missing(tmp_path, sparse):
    pgen = _write_pgen(tmp_path / "miss", _PSAM_IID)
    out = tmp_path / f"miss_{sparse}.zarr"
    returned = stream_pgen_to_zarr(pgen, str(out), sparse=sparse, genotype_encoding="int8", return_adata=True)

    expected = _GENO.T.astype(np.float32)
    expected[expected < 0] = np.nan
    for adata in (returned, read_pgen_zarr(out)):
        X = adata.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X.compute() if hasattr(X, "compute") else X)
        np.testing.assert_array_equal(X, expected)
        assert adata.uns["n_missing"] == 2

    if not sparse:  # the raw store keeps the compact int8 matrix with -1 for missing
        raw = np.asarray(read_pgen_zarr(out, backend="zarr").X[:])
        assert raw.dtype == np.int8
        np.testing.assert_array_equal(raw, np.where(_GENO.T < 0, -1, _GENO.T))


def test_without_missing_calls_x_stays_int8(tmp_path):
    out = tmp_path / "g.zarr"
    stream_pgen_to_zarr(str(DATA / "simulated_genotype_calls.pgen"), str(out))
    adata = read_pgen_zarr(out)
    assert adata.X.dtype == np.int8
    assert adata.uns["n_missing"] == 0


def test_multiple_inputs_need_the_same_sample_order(tmp_path):
    a = _write_pgen(tmp_path / "a", _PSAM_IID)
    b = _write_pgen(tmp_path / "b", _PSAM_IID)
    adata = stream_pgen_to_zarr([a, b], str(tmp_path / "ab.zarr"), return_adata=True)
    assert adata.shape == (5, 6)

    reordered = "#IID\tSEX\n" + "".join(f"S{i}\tNA\n" for i in reversed(range(5)))
    c = _write_pgen(tmp_path / "c", reordered)
    with pytest.raises(ValueError, match="same samples in the same order"):
        stream_pgen_to_zarr([a, c], str(tmp_path / "ac.zarr"))


def test_psam_sample_count_must_match_pgen(tmp_path):
    base = tmp_path / "short"
    for ext in ("pgen", "pvar"):
        shutil.copy(DATA / f"simulated_genotype_calls.{ext}", f"{base}.{ext}")
    Path(f"{base}.psam").write_text("#IID\tSEX\nS0\tNA\n")
    with pytest.raises(ValueError, match="lists 1 samples"):
        stream_pgen_to_zarr(f"{base}.pgen", str(tmp_path / "short.zarr"))
