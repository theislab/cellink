from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import zarr

from cellink._core import DonorData
from cellink.io import read_pgen_zarr, read_zarr_dd, write_zarr_dd
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
    dd = DonorData(D=G, C=C, donor_id="donor_id")
    donors, variants = [2, 40, 77], [1, 500, 998]
    np.testing.assert_array_equal(np.asarray(dd.sel(D_obs=donors).D.X), ref[donors])
    np.testing.assert_array_equal(np.asarray(dd.sel(D_var=variants).D.X), ref[:, variants])


@pytest.mark.slow
def test_write_zarr_dd_2bit_rounds_dosages(tmp_path, adata, gdata):
    gdata.X = np.clip(gdata.X + np.random.default_rng(0).uniform(-0.3, 0.3, gdata.X.shape), 0, 2)
    gdata.X[0, 0] = np.nan
    dd = DonorData(D=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    with pytest.warns(UserWarning, match="nearest integer"):
        write_zarr_dd(dd, str(out), x_chunks={"D": (gdata.n_obs, 4)})
    assert "cellink_pack2bit" in _codec_names(out / "D" / "X")
    expected = np.nan_to_num(np.rint(dd.D.X), nan=-1).astype(np.int8)
    np.testing.assert_array_equal(np.asarray(read_zarr_dd(str(out), backend="zarr").D.X), expected)
    # missing calls come back as nan by default
    np.testing.assert_array_equal(read_zarr_dd(str(out)).D.X.compute(), np.where(expected < 0, np.nan, expected))


@pytest.mark.slow
def test_write_zarr_dd_non_genotype_falls_back(tmp_path, adata, gdata):
    gdata.X = gdata.X.astype(float) * 5
    dd = DonorData(D=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    with pytest.warns(UserWarning, match="outside the 0-2"):
        write_zarr_dd(dd, str(out))
    assert "cellink_pack2bit" not in _codec_names(out / "D" / "X")
    np.testing.assert_array_equal(read_zarr_dd(str(out)).D.X.compute(), dd.D.X)


@pytest.mark.slow
def test_write_zarr_dd_without_encoding(tmp_path, adata, gdata):
    dd = DonorData(D=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    write_zarr_dd(dd, str(out), genotype_encoding=None)
    assert "cellink_pack2bit" not in _codec_names(out / "D" / "X")
    np.testing.assert_array_equal(read_zarr_dd(str(out)).D.X.compute(), dd.D.X)


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
    dd = DonorData(D=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    with pytest.warns(UserWarning, match="rounded up to 4"):
        write_zarr_dd(dd, str(out), x_chunks={"D": (2, 2), "C": (2, 2)})
    assert zarr.open_array(str(out / "D" / "X"), mode="r").chunks == (2, 4)
    np.testing.assert_array_equal(np.asarray(read_zarr_dd(str(out), backend="zarr").D.X), dd.D.X)


@pytest.mark.slow
def test_write_zarr_dd_2bit_keeps_missing_calls(tmp_path, adata, gdata):
    gdata.X = gdata.X.astype(np.int8)
    gdata.X[1, 3] = -9
    dd = DonorData(D=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    with pytest.warns(UserWarning, match="1 missing values"):
        write_zarr_dd(dd, str(out), x_chunks={"D": (gdata.n_obs, 4)})
    expected = np.where(dd.D.X < 0, -1, dd.D.X)
    np.testing.assert_array_equal(np.asarray(read_zarr_dd(str(out), backend="zarr").D.X), expected)
    X = read_zarr_dd(str(out)).D.X.compute()
    assert X.dtype == np.float32
    np.testing.assert_array_equal(X, np.where(expected < 0, np.nan, expected))
    assert zarr.open_array(str(out / "D" / "X"), mode="r").attrs["n_missing"] == 1


@pytest.mark.slow
def test_read_zarr_dd_keeps_int8_without_missing_calls(tmp_path, adata, gdata):
    gdata.X = gdata.X.astype(np.int8)
    dd = DonorData(D=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    write_zarr_dd(dd, str(out), x_chunks={"D": (gdata.n_obs, 4)})
    X = read_zarr_dd(str(out)).D.X.compute()
    assert X.dtype == np.int8
    np.testing.assert_array_equal(X, dd.D.X)
    # without the n_missing attribute the store may hold missing calls, so X is converted
    zarr.open_array(str(out / "D" / "X"), mode="r+").attrs.pop("n_missing")
    np.testing.assert_array_equal(read_zarr_dd(str(out)).D.X.compute(), dd.D.X.astype(np.float32))


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["dask", "zarr"])
def test_read_zarr_dd_loads_sparse_matrices(tmp_path, adata, gdata, backend):
    import scipy.sparse as sp

    adata.X = sp.csr_matrix(np.abs(adata.X))
    adata.layers["counts"] = sp.csc_matrix(adata.X)
    dd = DonorData(D=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    write_zarr_dd(dd, str(out), x_chunks={"D": (gdata.n_obs, 4)})
    loaded = read_zarr_dd(str(out), backend=backend)
    assert sp.issparse(loaded.C.X) and sp.issparse(loaded.C.layers["counts"])
    np.testing.assert_array_equal(loaded.C.X.toarray(), dd.C.X.toarray())
    np.testing.assert_array_equal(np.asarray(loaded.D.X), dd.D.X)


@pytest.mark.slow
def test_write_and_read_zarr_dd_with_samples_and_a_donor_table(tmp_path, adata, gdata):
    import pandas as pd

    gdata.X = gdata.X.astype(np.float32)
    gdata.X[0, 0] = np.nan
    adata.obs["sample_id"] = adata.obs["donor_id"].astype(str) + "_" + (np.arange(adata.n_obs) % 2).astype(str)
    samples = adata.obs[["sample_id", "donor_id"]].drop_duplicates().set_index("sample_id")
    samples["donor_id"] = samples["donor_id"].astype(str)
    samples["visit"] = np.arange(len(samples))
    dd = DonorData(D=gdata, S=samples, C=adata, sample_id="sample_id")
    out = tmp_path / "samples.dd.zarr"
    with pytest.warns(UserWarning, match="1 missing values"):
        write_zarr_dd(dd, str(out), x_chunks={"D": (gdata.n_obs, 4)})
    assert "cellink_pack2bit" in _codec_names(out / "D" / "X")
    loaded = read_zarr_dd(str(out))
    assert loaded.n_samples == dd.n_samples
    np.testing.assert_array_equal(loaded.S["visit"], dd.S["visit"])
    np.testing.assert_array_equal(loaded.D.X.compute(), dd.D.X)

    donors = pd.DataFrame({"age": np.arange(gdata.n_obs)}, index=gdata.obs_names)
    table = DonorData(D=donors, C=adata)
    write_zarr_dd(table, str(tmp_path / "table.dd.zarr"))
    loaded = read_zarr_dd(str(tmp_path / "table.dd.zarr"), backend="zarr")
    assert loaded.D_type == "table"
    np.testing.assert_array_equal(loaded.D["age"], donors["age"])
    np.testing.assert_array_equal(np.asarray(loaded.C.X), table.C.X)


@pytest.mark.slow
def test_read_zarr_dd_with_the_donor_level_in_g(tmp_path, adata, gdata):
    gdata.X = gdata.X.astype(np.int8)
    gdata.X[1, 3] = -1
    dd = DonorData(D=gdata, C=adata)
    out = tmp_path / "dd.dd.zarr"
    with pytest.warns(UserWarning, match="1 missing values"):
        write_zarr_dd(dd, str(out), x_chunks={"D": (gdata.n_obs, 4)})
    (out / "D").rename(out / "G")
    zarr.open_group(str(out), mode="r+").attrs["encoding-version"] = "0.1.0"
    with pytest.warns(UserWarning, match="donor level in G"):
        loaded = read_zarr_dd(str(out))
    X = loaded.D.X.compute()
    assert np.isnan(X[1, 3])
    np.testing.assert_array_equal(np.nan_to_num(X, nan=-1), dd.D.X)


@pytest.mark.slow
def test_write_and_read_zarr_dd_with_mudata_cells(tmp_path, adata, gdata):
    import mudata as md
    import scipy.sparse as sp

    atac = adata.copy()
    atac.X = sp.csr_matrix(np.abs(atac.X))
    mdata = md.MuData({"rna": adata, "atac": atac})
    mdata.obs["donor_id"] = adata.obs["donor_id"].reindex(mdata.obs_names)
    dd = DonorData(D=gdata, C=mdata)
    out = tmp_path / "dd.dd.zarr"
    write_zarr_dd(dd, str(out), x_chunks={"D": (gdata.n_obs, 4)})
    loaded = read_zarr_dd(str(out))
    assert sp.issparse(loaded.C.mod["atac"].X)
    np.testing.assert_array_equal(np.asarray(loaded.C.mod["rna"].X), dd.C.mod["rna"].X)
    np.testing.assert_array_equal(loaded.C.mod["atac"].X.toarray(), dd.C.mod["atac"].X.toarray())


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["dask", "zarr"])
def test_read_zarr_dd_keeps_dense_layers(tmp_path, adata, gdata, backend):
    gdata.X = np.asarray(gdata.X, dtype=np.float32)
    gdata.layers["dosage"] = gdata.X + 1
    out = tmp_path / "dd.dd.zarr"
    write_zarr_dd(DonorData(D=gdata, C=adata), str(out), x_chunks={"D": (gdata.n_obs, 4)})

    loaded = read_zarr_dd(str(out), backend=backend)
    assert [k for k in loaded.D.layers.keys() if k is not None] == ["dosage"]
    assert isinstance(loaded.D.layers["dosage"], GenotypeArray if backend == "zarr" else da.Array)
    np.testing.assert_array_equal(np.asarray(loaded.D.X), gdata.X)
    np.testing.assert_array_equal(np.asarray(loaded.D.layers["dosage"]), gdata.X + 1)


@pytest.mark.slow
def test_read_zarr_dd_rejects_a_store_whose_levels_do_not_match(tmp_path, adata, gdata):
    import anndata as ad

    out = tmp_path / "dd.dd.zarr"
    write_zarr_dd(DonorData(D=gdata, C=adata), str(out), x_chunks={"D": (gdata.n_obs, 4)})
    obs = adata.obs.copy()
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs.iloc[0, obs.columns.get_loc("donor_id")] = "nan"
    ad.io.write_elem(zarr.open_group(str(out / "C"), mode="r+"), "obs", obs)
    with pytest.raises(ValueError, match="do not match"):
        read_zarr_dd(str(out))


def test_write_zarr_dd_writes_genotypes_block_by_block(tmp_path, monkeypatch):
    import anndata as ad
    import pandas as pd

    x = np.random.default_rng(0).integers(0, 3, size=(6, 40), dtype=np.int8)
    z = zarr.create_array(
        store=str(tmp_path / "g.zarr"), shape=x.shape, chunks=(6, 8), dtype="int8", filters=[Pack2Bit()], fill_value=0
    )
    z[:] = x
    G = ad.AnnData(
        X=GenotypeArray.wrap(zarr.open_array(str(tmp_path / "g.zarr"), mode="r")),
        obs=pd.DataFrame(index=[f"d{i}" for i in range(6)]),
    )
    C = ad.AnnData(
        X=np.zeros((12, 1), dtype=np.float32),
        obs=pd.DataFrame({"donor_id": [f"d{i % 6}" for i in range(12)]}, index=[f"c{i}" for i in range(12)]),
    )
    dd = DonorData(D=G, C=C, donor_id="donor_id")

    def whole_matrix(self, dtype=None, copy=None):
        raise AssertionError("the whole genotype matrix was read into memory")

    monkeypatch.setattr(GenotypeArray, "__array__", whole_matrix)
    write_zarr_dd(dd, str(tmp_path / "dd.zarr"), x_chunks={"D": (6, 8)})
    monkeypatch.undo()
    assert "cellink_pack2bit" in _codec_names(tmp_path / "dd.zarr" / "D" / "X")
    np.testing.assert_array_equal(np.asarray(read_zarr_dd(str(tmp_path / "dd.zarr"), backend="zarr").D.X), x)
