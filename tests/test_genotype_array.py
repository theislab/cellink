from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import zarr

from cellink._core import DonorData
from cellink.io import read_pgen_zarr, read_zarr_dd
from cellink.io._codecs import Pack2Bit, to_genotype_codes
from cellink.io._genotype_array import GenotypeArray

DATA = Path("tests/data")


def _codec_names(path):
    return [c.to_dict()["name"] for c in zarr.open_array(path, mode="r").metadata.codecs]


def test_pack2bit_roundtrip(tmp_path):
    x = np.random.default_rng(0).integers(-1, 3, size=(37, 64), dtype=np.int8)
    z = zarr.create_array(
        store=str(tmp_path / "x.zarr"), shape=x.shape, chunks=(16, 32), dtype="int8", filters=[Pack2Bit()], fill_value=0
    )
    z[:] = x
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "x.zarr"), mode="r")[:], x)
    g = GenotypeArray.wrap(zarr.open_array(str(tmp_path / "x.zarr"), mode="r"))
    np.testing.assert_array_equal(g[[0, 5, 36], :], x[[0, 5, 36]])
    np.testing.assert_array_equal(g[:, [1, 2, 63]], x[:, [1, 2, 63]])
    with pytest.raises(ValueError, match="missing calls"):
        z[0, 0] = 3


def test_pack2bit_needs_chunks_divisible_by_4(tmp_path):
    with pytest.raises(ValueError, match="divisible by 4"):
        zarr.create_array(
            store=str(tmp_path / "x.zarr"), shape=(4, 10), chunks=(4, 10), dtype="int8", filters=[Pack2Bit()]
        )


def test_to_genotype_codes():
    codes, n_frac, n_missing = to_genotype_codes(np.array([[0, 1, 2, -9]], dtype=np.int8))
    np.testing.assert_array_equal(codes, [[0, 1, 2, -1]])
    assert (n_frac, n_missing) == (0, 1)

    codes, n_frac, n_missing = to_genotype_codes(np.array([[0.1, 0.9, 1.6, np.nan, -9.0]]), missing_value=-9)
    np.testing.assert_array_equal(codes, [[0, 1, 2, -1, -1]])
    assert (n_frac, n_missing) == (3, 2)

    with pytest.raises(ValueError):
        to_genotype_codes(np.array([[0.0, 3.0]]))


@pytest.mark.slow
@pytest.mark.parametrize("encoding", ["2bit", "int8"])
def test_stream_pgen_encodings(tmp_path, encoding):
    pgenlib = pytest.importorskip("pgenlib")
    from cellink.io import stream_pgen_to_zarr

    pgen_file = DATA / "simulated_genotype_calls.pgen"
    out = tmp_path / f"{encoding}.zarr"
    stream_pgen_to_zarr(str(pgen_file), str(out), chunk_samples=50, chunk_variants=200, genotype_encoding=encoding)
    assert ("cellink_pack2bit" in _codec_names(out / "X")) == (encoding == "2bit")

    expected = np.zeros((100, 1000), dtype=np.int8)
    pgenlib.PgenReader(bytes(str(pgen_file), "utf-8")).read_range(0, 1000, geno_int_out=expected, sample_maj=1)
    expected[expected < 0] = 0

    fast = read_pgen_zarr(str(out), backend="zarr")
    assert isinstance(fast.X, GenotypeArray)
    np.testing.assert_array_equal(np.asarray(fast.X), expected)
    lazy = read_pgen_zarr(str(out))
    assert isinstance(lazy.X, da.Array)
    np.testing.assert_array_equal(lazy.X.compute(), expected)


def test_stream_pgen_2bit_rounds_variant_chunks_up_to_4(tmp_path):
    pytest.importorskip("pgenlib")
    from cellink.io import stream_pgen_to_zarr

    out = tmp_path / "x.zarr"
    with pytest.warns(UserWarning, match="rounded up to 204"):
        stream_pgen_to_zarr(str(DATA / "simulated_genotype_calls.pgen"), str(out), chunk_variants=201)
    assert zarr.open_array(str(out / "X"), mode="r").chunks[1] == 204
    int8 = tmp_path / "int8.zarr"
    stream_pgen_to_zarr(str(DATA / "simulated_genotype_calls.pgen"), str(int8), chunk_variants=201, genotype_encoding="int8")
    np.testing.assert_array_equal(read_pgen_zarr(str(out)).X.compute(), read_pgen_zarr(str(int8)).X.compute())


@pytest.mark.slow
@pytest.mark.parametrize("encoding", ["2bit", "int8"])
def test_genotype_array_selections(tmp_path, encoding):
    pytest.importorskip("pgenlib")
    from cellink.io import stream_pgen_to_zarr

    out = tmp_path / f"{encoding}.zarr"
    stream_pgen_to_zarr(
        str(DATA / "simulated_genotype_calls.pgen"),
        str(out),
        chunk_samples=40,
        chunk_variants=96,
        genotype_encoding=encoding,
    )
    g = read_pgen_zarr(str(out), backend="zarr").X
    ref = zarr.open_array(str(out / "X"), mode="r")[:]
    rng = np.random.default_rng(1)
    rows, cols = rng.integers(0, 100, 17), rng.integers(0, 1000, 23)
    mask = rng.random(100) < 0.2
    for sel, exp in [
        (([3, 50, 99], slice(None)), ref[[3, 50, 99]]),
        ((slice(None), [0, 999]), ref[:, [0, 999]]),
        (np.ix_(rows, cols), ref[np.ix_(rows, cols)]),
        ((mask, slice(5, 900, 7)), ref[mask][:, 5:900:7]),
        ((slice(10, 90), slice(950, None)), ref[10:90, 950:]),
        ((slice(None), slice(None)), ref),
    ]:
        np.testing.assert_array_equal(g[sel], exp)
    assert g[5, 7] == ref[5, 7]
    np.testing.assert_array_equal(g[5], ref[5])


@pytest.mark.slow
def test_donordata_sel_on_genotype_array(tmp_path, adata):
    pytest.importorskip("pgenlib")
    import anndata as ad

    from cellink.io import stream_pgen_to_zarr

    out = tmp_path / "g.zarr"
    stream_pgen_to_zarr(str(DATA / "simulated_genotype_calls.pgen"), str(out), chunk_samples=50, chunk_variants=200)
    G = read_pgen_zarr(str(out), backend="zarr")
    ref = zarr.open_array(str(out / "X"), mode="r")[:]
    C = ad.AnnData(X=np.zeros((G.n_obs, 1)))
    C.obs["donor_id"] = G.obs_names.values
    G.obs["donor_id"] = G.obs_names.values
    dd = DonorData(G=G, C=C, donor_id="donor_id")
    donors, variants = [2, 40, 77], [1, 500, 998]
    np.testing.assert_array_equal(np.asarray(dd.sel(G_obs=donors).G.X), ref[donors])
    np.testing.assert_array_equal(np.asarray(dd.sel(G_var=variants).G.X), ref[:, variants])


@pytest.mark.slow
def test_write_zarr_dd_2bit_rounds_dosages(tmp_path, adata, gdata):
    gdata.X = np.clip(gdata.X + np.random.default_rng(0).uniform(-0.3, 0.3, gdata.X.shape), 0, 2)
    gdata.X[0, 0] = np.nan
    dd = DonorData(G=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    with pytest.warns(UserWarning, match="nearest integer"):
        dd.write_zarr_dd(str(out), x_chunks={"G": (gdata.n_obs, 4)})
    assert "cellink_pack2bit" in _codec_names(out / "G" / "X")
    expected = np.nan_to_num(np.rint(dd.G.X), nan=-1).astype(np.int8)
    np.testing.assert_array_equal(np.asarray(read_zarr_dd(str(out), backend="zarr").G.X), expected)
    # missing calls come back as nan by default
    np.testing.assert_array_equal(read_zarr_dd(str(out)).G.X.compute(), np.where(expected < 0, np.nan, expected))


@pytest.mark.slow
def test_write_zarr_dd_non_genotype_falls_back(tmp_path, adata, gdata):
    gdata.X = gdata.X.astype(float) * 5
    dd = DonorData(G=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    with pytest.warns(UserWarning, match="outside the 0-2"):
        dd.write_zarr_dd(str(out))
    assert "cellink_pack2bit" not in _codec_names(out / "G" / "X")
    np.testing.assert_array_equal(read_zarr_dd(str(out)).G.X.compute(), dd.G.X)


@pytest.mark.slow
def test_write_zarr_dd_without_encoding(tmp_path, adata, gdata):
    dd = DonorData(G=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    dd.write_zarr_dd(str(out), genotype_encoding=None)
    assert "cellink_pack2bit" not in _codec_names(out / "G" / "X")
    np.testing.assert_array_equal(read_zarr_dd(str(out)).G.X.compute(), dd.G.X)


def _write_dosage_pgen(prefix, dosages):
    pgenlib = pytest.importorskip("pgenlib")
    n_var, n_samp = dosages.shape
    w = pgenlib.PgenWriter(f"{prefix}.pgen".encode(), n_samp, variant_ct=n_var, nonref_flags=False, dosage_present=True)
    for d in dosages:
        w.append_dosages(d)
    w.close()
    Path(f"{prefix}.psam").write_text("#IID\tSEX\n" + "".join(f"S{i}\tNA\n" for i in range(n_samp)))
    Path(f"{prefix}.pvar").write_text(
        "#CHROM\tPOS\tID\tREF\tALT\n" + "".join(f"1\t{i + 1}\tv{i}\tA\tG\n" for i in range(n_var))
    )


@pytest.mark.slow
def test_stream_pgen_2bit_dosages(tmp_path):
    from cellink.io import stream_pgen_to_zarr

    dosages = np.array(
        [[0, 1, 2, 0.6, 1.45, -9], [0, 0.95, 1.2, 2, 1, 0], [1, 1, 1, 1, 1, 1], [0.3, 1.7, 0, 2, 0.4, 1.6]] * 2,
        dtype=np.float32,
    )
    prefix = tmp_path / "dos"
    _write_dosage_pgen(prefix, dosages)
    with pytest.warns(UserWarning, match="imputed dosages"):
        stream_pgen_to_zarr(str(prefix), str(tmp_path / "2bit.zarr"), chunk_variants=4)
    two_bit = read_pgen_zarr(str(tmp_path / "2bit.zarr"), backend="zarr")
    stored = np.asarray(two_bit.X)
    np.testing.assert_array_equal(stored, np.where(dosages == -9, -1, np.rint(dosages)).T.astype(np.int8))
    assert two_bit.uns["n_missing"] == (dosages == -9).sum()

    stream_pgen_to_zarr(str(prefix), str(tmp_path / "int8.zarr"), chunk_variants=4, genotype_encoding="int8")
    hard = np.asarray(read_pgen_zarr(str(tmp_path / "int8.zarr"), backend="zarr").X)
    assert (hard != stored).any()


@pytest.mark.slow
def test_write_zarr_dd_2bit_rounds_variant_chunks_up_to_4(tmp_path, adata, gdata):
    dd = DonorData(G=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    with pytest.warns(UserWarning, match="rounded up to 4"):
        dd.write_zarr_dd(str(out), x_chunks={"G": (2, 2), "C": (2, 2)})
    assert zarr.open_array(str(out / "G" / "X"), mode="r").chunks == (2, 4)
    np.testing.assert_array_equal(np.asarray(read_zarr_dd(str(out), backend="zarr").G.X), dd.G.X)


@pytest.mark.slow
def test_write_zarr_dd_2bit_keeps_missing_calls(tmp_path, adata, gdata):
    gdata.X = gdata.X.astype(np.int8)
    gdata.X[1, 3] = -9
    dd = DonorData(G=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    with pytest.warns(UserWarning, match="1 missing values"):
        dd.write_zarr_dd(str(out), x_chunks={"G": (gdata.n_obs, 4)})
    expected = np.where(dd.G.X < 0, -1, dd.G.X)
    np.testing.assert_array_equal(np.asarray(read_zarr_dd(str(out), backend="zarr").G.X), expected)
    X = read_zarr_dd(str(out)).G.X.compute()
    assert X.dtype == np.float32
    np.testing.assert_array_equal(X, np.where(expected < 0, np.nan, expected))
    assert zarr.open_array(str(out / "G" / "X"), mode="r").attrs["n_missing"] == 1


@pytest.mark.slow
def test_read_zarr_dd_keeps_int8_without_missing_calls(tmp_path, adata, gdata):
    gdata.X = gdata.X.astype(np.int8)
    dd = DonorData(G=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    dd.write_zarr_dd(str(out), x_chunks={"G": (gdata.n_obs, 4)})
    X = read_zarr_dd(str(out)).G.X.compute()
    assert X.dtype == np.int8
    np.testing.assert_array_equal(X, dd.G.X)
    # without the n_missing attribute the store may hold missing calls, so X is converted
    zarr.open_array(str(out / "G" / "X"), mode="r+").attrs.pop("n_missing")
    np.testing.assert_array_equal(read_zarr_dd(str(out)).G.X.compute(), dd.G.X.astype(np.float32))
