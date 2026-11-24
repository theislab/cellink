import importlib.util
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellink import HiData
from cellink.io import from_sgkit_dataset, read_h5_hd, read_plink, read_sgkit_zarr, to_pgen, to_plink, write_h5_hd

DATA = Path("tests/data")


@pytest.mark.slow
def test_read_plink():
    pytest.importorskip("bed_reader")
    gdata = read_plink(DATA / "simulated_genotype_calls")
    assert gdata.shape == (100, 1000)
    X = np.asarray(gdata.X.compute())
    assert set(np.unique(X)) <= {0, 1, 2}
    assert list(gdata.var.columns) == ["contig", "pos", "id", "a0", "a1", "chrom"]
    assert list(gdata.obs.columns) == ["family_id", "member_id", "paternal_id", "maternal_id", "sex", "phenotype"]


@pytest.mark.parametrize(
    ("fam_sep", "bim_sep", "pad"), [("\t", "\t", ""), (" ", "\t", ""), (" ", " ", ""), ("  ", " \t ", "  ")]
)
def test_read_plink_any_whitespace_and_unknown_chromosome(tmp_path, fam_sep, bim_sep, pad):
    pytest.importorskip("bed_reader")
    src = DATA / "simulated_genotype_calls"
    expected = read_plink(src)
    (tmp_path / "ws.bed").write_bytes(Path(f"{src}.bed").read_bytes())
    for ext, sep in (("fam", fam_sep), ("bim", bim_sep)):
        rows = [line.split() for line in Path(f"{src}.{ext}").read_text().splitlines()]
        (tmp_path / f"ws.{ext}").write_text("".join(pad + sep.join(r) + "\n" for r in rows))
    got = read_plink(tmp_path / "ws.bed")
    np.testing.assert_array_equal(np.asarray(got.X.compute()), np.asarray(expected.X.compute()))
    assert got.obs.astype(str).equals(expected.obs.astype(str))
    assert got.var.astype(str).equals(expected.var.astype(str))
    assert list(got.obs.index[:2]) == ["S0", "S1"]
    assert set(got.var["chrom"]) == {"0"}  # PLINK's code for an unknown chromosome
    assert got.var.index[0].startswith("0_")


def test_read_plink_count_a1():
    pytest.importorskip("bed_reader")
    col6 = read_plink(DATA / "simulated_genotype_calls", count_a1=False)
    default = read_plink(DATA / "simulated_genotype_calls")
    np.testing.assert_array_equal(np.asarray(default.X.compute()), 2 - np.asarray(col6.X.compute()))
    assert list(default.var.a0) == list(col6.var.a1) and list(default.var.a1) == list(col6.var.a0)
    var = default.var
    assert list(var.index) == [
        f"{c}_{p}_{a0}_{a1}" for c, p, a0, a1 in zip(var.chrom, var.pos, var.a0, var.a1, strict=True)
    ]
    assert default.obs.equals(col6.obs)
    for k in (0, 1):
        old = np.asarray(col6.layers[f"GENOTYPE_{k}"].compute())
        np.testing.assert_array_equal(
            np.asarray(default.layers[f"GENOTYPE_{k}"].compute()), np.where(old >= 0, 1 - old, old)
        )
        np.testing.assert_array_equal(
            np.asarray(default.layers[f"GENOTYPE_MASK_{k}"].compute()),
            np.asarray(col6.layers[f"GENOTYPE_MASK_{k}"].compute()),
        )


def test_call_fields_follow_selections():
    pytest.importorskip("bed_reader")
    # per-call fields are layers shaped like X, so subsetting samples and variants subsets them too
    gdata = read_plink(DATA / "simulated_genotype_calls")
    assert not any(str(k).startswith("raw_call") for k in gdata.uns)
    assert not any(hasattr(v, "ndim") and getattr(v, "ndim", 0) >= 2 for v in gdata.uns.values())
    sub = gdata[[0, 2, 5], 1:4]
    for key in ("GENOTYPE_0", "GENOTYPE_1", "GENOTYPE_MASK_0", "GENOTYPE_MASK_1"):
        assert sub.layers[key].shape == sub.shape
        np.testing.assert_array_equal(np.asarray(sub.layers[key]), np.asarray(gdata.layers[key])[[0, 2, 5]][:, 1:4])
    # the two allele copies add up to the ALT dosage in X (read_plink counts a1 = ALT by default)
    alt = np.asarray(sub.layers["GENOTYPE_0"]) + np.asarray(sub.layers["GENOTYPE_1"])
    called = ~(np.asarray(sub.layers["GENOTYPE_MASK_0"]) | np.asarray(sub.layers["GENOTYPE_MASK_1"]))
    np.testing.assert_array_equal(alt[called], np.asarray(sub.X)[called])


# PLINK files to_plink writes for _PLINK_X below (a0 = REF = "A", a1 = ALT = "G"): ALT "G" is in
# .bim column 5 and the .bed codes count it (00 = two copies of G, 10 = one, 11 = none, 01 = missing).
_PLINK_X = np.array(
    [
        [0, 1, 2, np.nan, 2],
        [2, 2, 0, 0, 1],
        [1, np.nan, 0, 2, 0],
        [0, 0, 1, 1, 2],
        [2, 1, np.nan, 0, 0],
    ],
    dtype="float32",
)
_PLINK_BED = bytes.fromhex("6c1b01e300d202bc018d033803")
_PLINK_BIM = "".join(f"1\t1_{100 + i}_A_G\t0.0\t{100 + i}\tG\tA\n" for i in range(5))
_PLINK_FAM = "".join(f"D{i}\tD{i}\t0\t0\t{s}\t-0\n" for i, s in enumerate([1, 2, 0, 1, 2]))


def _plink_gdata(X=_PLINK_X):
    import anndata as ad
    import pandas as pd

    n_obs, n_vars = X.shape
    obs = pd.DataFrame({"donor_id": [f"D{i}" for i in range(n_obs)], "sex": ([1, 2, 0, 1, 2] * n_obs)[:n_obs]})
    obs.index = obs["donor_id"].to_numpy()
    var = pd.DataFrame(
        {"chrom": ["1"] * n_vars, "pos": [100 + i for i in range(n_vars)], "a0": ["A"] * n_vars, "a1": ["G"] * n_vars},
        index=[f"1_{100 + i}_A_G" for i in range(n_vars)],
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def test_to_plink_bytes(tmp_path):
    to_plink(_plink_gdata(), output_prefix=str(tmp_path / "out"))
    assert (tmp_path / "out.bed").read_bytes() == _PLINK_BED
    assert (tmp_path / "out.bim").read_text() == _PLINK_BIM
    assert (tmp_path / "out.fam").read_text() == _PLINK_FAM


def test_export_writes_unknown_sex_as_0(tmp_path):
    gdata = _plink_gdata()
    gdata.obs["sex"] = [1.0, np.nan, 2.0, -1.0, 0.0]
    to_plink(gdata, output_prefix=str(tmp_path / "s"))
    fam = [line.split("\t") for line in (tmp_path / "s.fam").read_text().splitlines()]
    assert [r[4] for r in fam] == ["1", "0", "2", "0", "0"]


def test_export_reads_negative_integers_as_missing(tmp_path):
    pgenlib = pytest.importorskip("pgenlib")
    as_int8 = np.where(np.isnan(_PLINK_X), -1, _PLINK_X).astype(np.int8)
    to_plink(_plink_gdata(), output_prefix=str(tmp_path / "f"))
    to_plink(_plink_gdata(as_int8), output_prefix=str(tmp_path / "i"))
    assert (tmp_path / "f.bed").read_bytes() == (tmp_path / "i.bed").read_bytes()
    to_pgen(_plink_gdata(as_int8), str(tmp_path / "i"))
    reader = pgenlib.PgenReader(str(tmp_path / "i.pgen").encode())
    calls = np.empty((5, 5), dtype=np.int8)
    reader.read_range(0, 5, calls)
    np.testing.assert_array_equal(calls.T, np.where(as_int8 < 0, -9, as_int8))  # pgenlib reads missing as -9


def test_to_plink_accepts_bed_suffix_and_dask_and_sparse(tmp_path):
    import dask.array as da
    import scipy.sparse as sp

    for i, X in enumerate([da.from_array(_PLINK_X, chunks=(2, 2)), sp.csr_matrix(np.nan_to_num(_PLINK_X))]):
        to_plink(_plink_gdata(X), output_prefix=str(tmp_path / f"x{i}.bed"), variant_chunk_size=2)
        expected = _plink_gdata(np.asarray(X.compute() if hasattr(X, "compute") else X.toarray()))
        to_plink(expected, output_prefix=str(tmp_path / f"ref{i}"))
        assert (tmp_path / f"x{i}.bed").read_bytes() == (tmp_path / f"ref{i}.bed").read_bytes()


@pytest.mark.parametrize("n_samples", [1, 2, 3, 4, 5, 7, 8, 13])
def test_plink_roundtrip(tmp_path, n_samples):
    pytest.importorskip("bed_reader")
    rng = np.random.default_rng(n_samples)
    X = rng.integers(0, 3, size=(n_samples, 11)).astype("float32")
    X[rng.random(X.shape) < 0.15] = np.nan
    to_plink(_plink_gdata(X), output_prefix=str(tmp_path / "rt"))
    back = read_plink(tmp_path / "rt")
    np.testing.assert_array_equal(np.asarray(back.X.compute()), X)
    assert list(back.obs.index) == [f"D{i}" for i in range(n_samples)]
    assert list(back.var.a0) == ["A"] * 11 and list(back.var.a1) == ["G"] * 11
    assert list(back.var.chrom) == ["1"] * 11


def test_to_plink_hard_calls_only(tmp_path):
    pytest.importorskip("bed_reader")
    X = _PLINK_X.copy()
    X[0, 0] = 1.6
    with pytest.warns(UserWarning, match="rounded"):
        to_plink(_plink_gdata(X), output_prefix=str(tmp_path / "rounded"))
    assert np.asarray(read_plink(tmp_path / "rounded").X.compute())[0, 0] == 2
    X[0, 0] = 3
    with pytest.raises(ValueError, match="hard calls"):
        to_plink(_plink_gdata(X), output_prefix=str(tmp_path / "bad"))


def test_to_plink_parent_ids(tmp_path):
    gdata = _plink_gdata()
    gdata.obs["donor_paternal_id"] = [f"P{i}" for i in range(5)]
    gdata.obs["donor_maternal_id"] = [f"M{i}" for i in range(5)]
    to_plink(gdata, output_prefix=str(tmp_path / "fam"))
    fam = [line.split("\t") for line in (tmp_path / "fam.fam").read_text().splitlines()]
    assert [r[2] for r in fam] == [f"P{i}" for i in range(5)]
    assert [r[3] for r in fam] == [f"M{i}" for i in range(5)]


"""
@pytest.mark.slow
def test_read_bgen():
    read_bgen(DATA / "simulated_genotype_calls")
"""


@pytest.mark.slow
def test_read_sgkit_zarr():
    pytest.importorskip("sgkit")
    read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")


@pytest.mark.slow
def test_from_plink_sgkit_dataset():
    pytest.importorskip("bed_reader")
    from sgkit.io.plink import read_plink as sg_read_plink

    gdata = from_sgkit_dataset(sg_read_plink(path=DATA / "simulated_genotype_calls"))
    assert gdata.shape == (100, 1000)


@pytest.mark.slow
def test_read_bgen():
    pytest.importorskip("sgkit")
    pytest.importorskip("cbgen")  # sgkit's BGEN backend, not a cellink dependency
    from cellink.io import read_bgen

    gdata = read_bgen(DATA / "simulated_genotype_calls.bgen", X_field="DS")
    assert gdata.shape == (100, 1000)


@pytest.mark.slow
def test_export(tmp_path):
    pytest.importorskip("sgkit")
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    gdata = gdata[:, :1996]
    gdata.obs["donor_id"] = gdata.obs.index

    to_plink(gdata, output_prefix=str(tmp_path))


def _dense(x):
    x = x.compute() if hasattr(x, "compute") else x
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x)


@pytest.mark.parametrize("chunks", [None, (3, 5)])
def test_read_h5_hd_lazy(tmp_path, adata, gdata, chunks):
    import dask.array as da
    import h5py
    import scipy.sparse as sp

    from cellink.io import read_hd

    X = np.asarray(gdata.X, dtype=np.float32)
    X[0, 1] = np.nan
    gdata.X = X
    gdata.layers["dosage"] = X / 2
    gdata.obs["donor_id"] = gdata.obs.index
    adata.X = sp.csr_matrix(adata.X)
    path = str(tmp_path / "hidata.hd.h5")
    write_h5_hd(HiData(G=gdata, C=adata, group_id="donor_id"), path)
    if chunks is not None:
        with h5py.File(path, "r+") as f:
            attrs, values = dict(f["G/X"].attrs), f["G/X"][()]
            del f["G/X"]
            f.create_dataset("G/X", data=values, chunks=chunks)
            f["G/X"].attrs.update(attrs)

    eager = read_h5_hd(path)
    lazy = read_h5_hd(path, lazy=True)
    assert isinstance(lazy.G.X, da.Array)
    assert isinstance(lazy.G.layers["dosage"], da.Array)
    assert lazy.G.X.chunksize == (chunks or lazy.G.shape)
    assert sp.issparse(lazy.C.X)
    for name in ("G", "C"):
        a, b = getattr(lazy, name), getattr(eager, name)
        np.testing.assert_array_equal(_dense(a.X), _dense(b.X))
        pd.testing.assert_frame_equal(a.obs, b.obs)
        pd.testing.assert_frame_equal(a.var, b.var)
    np.testing.assert_array_equal(_dense(lazy.G.layers["dosage"]), eager.G.layers["dosage"])
    np.testing.assert_array_equal(_dense(lazy.G[1:4, 2:9].X), eager.G.X[1:4, 2:9])
    assert isinstance(read_hd(path, lazy=True).G.X, da.Array)
    assert not isinstance(read_hd(path).G.X, da.Array)

    to_plink(lazy.G, output_prefix=str(tmp_path / "lazy"))
    to_plink(eager.G, output_prefix=str(tmp_path / "eager"))
    for ext in ("bed", "bim", "fam"):
        assert (tmp_path / f"lazy.{ext}").read_bytes() == (tmp_path / f"eager.{ext}").read_bytes()
    if importlib.util.find_spec("pgenlib") is not None:
        to_pgen(lazy.G, str(tmp_path / "lazy"))
        to_pgen(eager.G, str(tmp_path / "eager"))
        for ext in ("pgen", "pvar", "psam"):
            assert (tmp_path / f"lazy.{ext}").read_bytes() == (tmp_path / f"eager.{ext}").read_bytes()


@pytest.mark.parametrize("level_key", ["D", "G"])
@pytest.mark.parametrize("lazy", [False, True])
def test_read_h5_hd_reads_donordata_files(tmp_path, adata, gdata, level_key, lazy):
    import warnings

    import h5py

    path = str(tmp_path / "cohort.dd.h5")
    write_h5_hd(HiData(G=gdata, C=adata, group_id="donor_id"), path)
    with h5py.File(path, "r+") as f:
        if level_key == "D":
            f.move("G", "D")
        f.attrs["encoding-type"] = "donordata"
        f.attrs["encoding-version"] = "0.3.0" if level_key == "D" else "0.1.0"
        f.attrs["donor_id"] = f.attrs["group_id"]
        del f.attrs["group_id"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = read_h5_hd(path, lazy=lazy)
    assert loaded.group_id == "donor_id"
    np.testing.assert_array_equal(_dense(loaded.G.X), gdata.X)
    np.testing.assert_array_equal(_dense(loaded.C.X), _dense(adata.X))
    pd.testing.assert_frame_equal(loaded.C.obs, adata.obs.loc[loaded.C.obs_names], check_categorical=False)


@pytest.mark.slow
def test_stream_pgen_to_zarr_roundtrip(tmp_path):
    pgenlib = pytest.importorskip("pgenlib")

    from cellink.io import read_pgen_zarr, stream_pgen_to_zarr

    pgen_file = DATA / "simulated_genotype_calls.pgen"
    out_path = tmp_path / "pgen.zarr"

    result = stream_pgen_to_zarr(str(pgen_file), str(out_path), chunk_samples=50, chunk_variants=200)
    assert result is None  # return_adata defaults to False, this is normally a one-off conversion

    reloaded = read_pgen_zarr(str(out_path))
    X = np.asarray(reloaded.X.compute())
    assert X.shape == (100, 1000)

    # cross-check against pgenlib's own reader, independent of cellink's streaming path
    reader = pgenlib.PgenReader(bytes(str(pgen_file), "utf-8"))
    expected = np.zeros((100, 1000), dtype=np.int8)
    reader.read_range(0, 1000, geno_int_out=expected, sample_maj=1)
    np.testing.assert_array_equal(X, expected)


def test_stream_pgen_to_zarr_writes_anndata_encoding(tmp_path):
    """The dense store is a regular AnnData Zarr store: anndata reads it without old-format warnings."""
    pytest.importorskip("pgenlib")
    from anndata import OldFormatWarning

    from cellink.io import stream_pgen_to_zarr

    out_path = tmp_path / "pgen.zarr"
    stream_pgen_to_zarr(str(DATA / "simulated_genotype_calls.pgen"), str(out_path), chunk_variants=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error", OldFormatWarning)
        reloaded = ad.read_zarr(out_path)
    assert reloaded.shape == (100, 1000)
    assert dict(reloaded.uns) == {"n_missing": 0}  # no missing calls in this file
    assert len(reloaded.obsm) == len(reloaded.varm) == 0


@pytest.mark.slow
def test_stream_pgen_to_zarr_no_compression(tmp_path):
    """compressor=None writes a plain, uncompressed Zarr array -- the fair
    comparison point against PGEN's own storage_mode=0x10 (fixed 2-bit, no
    compression, no difference lists): both stores should decode to the same
    genotypes, and skipping BloscCodec entirely (not clevel=0, which still
    pays codec framing overhead) should not change any value.
    """
    pytest.importorskip("pgenlib")
    from cellink.io import read_pgen_zarr, stream_pgen_to_zarr

    pgen_file = DATA / "simulated_genotype_calls.pgen"
    out_path = tmp_path / "pgen_nocompression.zarr"

    stream_pgen_to_zarr(str(pgen_file), str(out_path), chunk_samples=50, chunk_variants=200, compressor=None)
    reloaded = read_pgen_zarr(str(out_path), backend="zarr")
    assert reloaded.X.compressors == ()
    X = np.asarray(reloaded.X[:])

    out_path_compressed = tmp_path / "pgen_compressed.zarr"
    stream_pgen_to_zarr(str(pgen_file), str(out_path_compressed), chunk_samples=50, chunk_variants=200)
    X_compressed = np.asarray(read_pgen_zarr(str(out_path_compressed), backend="zarr").X[:])
    np.testing.assert_array_equal(X, X_compressed)


@pytest.mark.slow
def test_read_pgen_zarr_var_columns(tmp_path):
    pytest.importorskip("pgenlib")
    import pandas as pd

    from cellink.io import read_pgen_zarr, stream_pgen_to_zarr

    out_path = tmp_path / "pgen.zarr"
    stream_pgen_to_zarr(
        str(DATA / "simulated_genotype_calls.pgen"), str(out_path), chunk_samples=50, chunk_variants=200
    )
    full = read_pgen_zarr(str(out_path))
    lean = read_pgen_zarr(str(out_path), var_columns=["chrom", "pos"])
    assert list(lean.var.columns) == ["chrom", "pos"]
    pd.testing.assert_frame_equal(lean.var, full.var[["chrom", "pos"]])

    bare = read_pgen_zarr(str(out_path), backend="zarr", var_columns=[])
    assert bare.var.shape == (1000, 0)
    pd.testing.assert_index_equal(bare.var_names, full.var_names)
    np.testing.assert_array_equal(np.asarray(bare.X[:]), np.asarray(full.X.compute()))


@pytest.mark.slow
def test_stream_pgen_to_zarr_sparse(tmp_path):
    pytest.importorskip("pgenlib")

    from cellink.io import stream_pgen_to_zarr

    pgen_file = DATA / "simulated_genotype_calls.pgen"
    out_path = tmp_path / "pgen_sparse.zarr"

    result = stream_pgen_to_zarr(str(pgen_file), str(out_path), sparse=True)
    assert result is None

    adata = stream_pgen_to_zarr(str(pgen_file), str(out_path), sparse=True, return_adata=True)
    assert adata.shape == (100, 1000)


def _pgen_gdata(X):
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame({"sex": [1, 2, 0]}, index=["S0", "S1", "S2"]),
        var=pd.DataFrame(
            {"chrom": ["1", "1", "2", "2"], "pos": [10, 20, 30, 40], "a0": list("AACG"), "a1": list("GTTA")},
            index=["v0", "v1", "v2", "v3"],
        ),
    )


def _read_pgen(prefix, n_variants=4, n_samples=3):
    pgenlib = pytest.importorskip("pgenlib")
    reader = pgenlib.PgenReader(f"{prefix}.pgen".encode())
    assert (reader.get_raw_sample_ct(), reader.get_variant_ct()) == (n_samples, n_variants)
    dosages = np.empty((n_variants, n_samples), dtype=np.float32)
    reader.read_dosages_range(0, n_variants, dosages)
    calls = np.empty((n_variants, n_samples), dtype=np.int8)
    reader.read_range(0, n_variants, calls)
    return dosages, calls


# samples x variants; v1 holds a dosage, v0 and v2 a missing call
_PGEN_X = np.array([[0, 1, 2, np.nan], [2, 1.5, 0, 1], [1, 0, np.nan, 2]])


def test_to_pgen_writes_calls_dosages_and_tables(tmp_path):
    pytest.importorskip("pgenlib")
    to_pgen(_pgen_gdata(_PGEN_X), tmp_path / "out.pgen")
    dosages, calls = _read_pgen(tmp_path / "out")
    expected = np.where(np.isnan(_PGEN_X), -9, _PGEN_X).T  # pgenlib reports missing as -9
    np.testing.assert_allclose(dosages, expected, atol=1 / 16384)
    np.testing.assert_array_equal(calls[[0, 2, 3]], expected[[0, 2, 3]])  # hard-call variants

    pvar = pd.read_csv(tmp_path / "out.pvar", sep="\t", dtype=str)
    assert pvar.to_dict("list") == {
        "#CHROM": ["1", "1", "2", "2"],
        "POS": ["10", "20", "30", "40"],
        "ID": ["v0", "v1", "v2", "v3"],
        "REF": list("AACG"),
        "ALT": list("GTTA"),
    }
    psam = pd.read_csv(tmp_path / "out.psam", sep="\t", dtype=str, keep_default_na=False)
    assert psam.to_dict("list") == {"#IID": ["S0", "S1", "S2"], "SEX": ["1", "2", "NA"]}


def test_to_pgen_hard_calls_only_or_dosages_only(tmp_path):
    pytest.importorskip("pgenlib")
    with pytest.warns(UserWarning, match="rounded to the nearest hard call"):
        to_pgen(_pgen_gdata(_PGEN_X), tmp_path / "calls", dosages=False)
    dosages, calls = _read_pgen(tmp_path / "calls")
    rounded = np.where(np.isnan(_PGEN_X), -9, np.rint(_PGEN_X)).T
    np.testing.assert_array_equal(calls, rounded)
    np.testing.assert_array_equal(dosages, rounded)

    to_pgen(_pgen_gdata(_PGEN_X), tmp_path / "dosages", dosages=True)
    dosages, _ = _read_pgen(tmp_path / "dosages")
    np.testing.assert_allclose(dosages, np.where(np.isnan(_PGEN_X), -9, _PGEN_X).T, atol=1 / 16384)


def test_to_pgen_rejects_values_outside_0_2_and_leaves_no_files(tmp_path):
    pytest.importorskip("pgenlib")
    X = _PGEN_X.copy()
    X[0, 3] = 3
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        to_pgen(_pgen_gdata(X), tmp_path / "bad")
    assert not list(tmp_path.iterdir())
