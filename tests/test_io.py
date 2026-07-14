from pathlib import Path

import numpy as np
import pytest

from cellink import DonorData
from cellink.io import from_sgkit_dataset, read_h5_dd, read_plink, read_sgkit_zarr, read_zarr_dd, to_plink

DATA = Path("tests/data")


@pytest.mark.slow
def test_read_plink():
    pytest.importorskip("sgkit")
    gdata = read_plink(DATA / "simulated_genotype_calls")
    assert gdata.shape == (100, 1000)


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
    pytest.importorskip("sgkit")
    from sgkit.io.plink import read_plink as sg_read_plink

    sgkit_dataset = sg_read_plink(path=DATA / "simulated_genotype_calls")
    from_sgkit_dataset(sgkit_dataset)


"""
@pytest.mark.slow
def test_from_bgen_sgkit_dataset():
    from sgkit.io.bgen import read_bgen as sg_read_bgen

    sgkit_dataset = sg_read_bgen(path=DATA / "simulated_genotype_calls")
    from_sgkit_dataset(sgkit_dataset)
"""


@pytest.mark.slow
def test_export(tmp_path):
    pytest.importorskip("sgkit")
    gdata = read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")
    gdata = gdata[:, :1996]
    gdata.obs["donor_id"] = gdata.obs.index

    to_plink(gdata, output_prefix=str(tmp_path))


@pytest.mark.slow
def test_read_h5_dd(tmp_path, adata, gdata):
    output_path = tmp_path / "donordata.dd.h5"

    dd = DonorData(G=gdata, C=adata)
    dd.write_h5_dd(str(output_path))

    dd_loaded = read_h5_dd(str(output_path))

    assert dd_loaded.C.shape == dd.C.shape
    assert dd_loaded.G.shape == dd.G.shape


@pytest.mark.slow
def test_read_zarr_dd(tmp_path, adata, gdata):
    output_path = tmp_path / "donordata.dd.zarr"

    dd = DonorData(G=gdata, C=adata)
    dd.write_zarr_dd(str(output_path))

    dd_loaded = read_zarr_dd(str(output_path))

    assert dd_loaded.C.shape == dd.C.shape
    assert dd_loaded.G.shape == dd.G.shape


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
