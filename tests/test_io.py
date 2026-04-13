from pathlib import Path

import pytest

from cellink import DonorData
from cellink.io import from_sgkit_dataset, read_bgen, read_h5_dd, read_plink, read_sgkit_zarr, read_zarr_dd, to_plink, write_variants_to_vcf
from anndata.experimental import read_lazy
from anndata import read_zarr
DATA = Path("tests/data")


@pytest.mark.slow
def test_read_plink():
    read_plink(DATA / "simulated_genotype_calls")

"""
@pytest.mark.slow
def test_read_bgen():
    read_bgen(DATA / "simulated_genotype_calls")
"""

@pytest.mark.slow
def test_read_sgkit_zarr():
    read_sgkit_zarr(DATA / "simulated_genotype_calls.vcz")


@pytest.mark.slow
def test_from_plink_sgkit_dataset():
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
def test_export_vcf(tmp_path, gdata, adata):
    output_path_zarr = tmp_path / "donordata.dd.zarr"
    output_path_vcf = tmp_path / "exported.vcf"
    output_path_vcf_lazy = tmp_path / "exported_lazy.vcf"
    gdata.write_zarr(output_path_zarr)
    gdata_loaded = read_zarr(output_path_zarr)
    gdata_loaded_lazy = read_lazy(output_path_zarr)
    dd_loaded = DonorData(G=gdata_loaded, C=adata)
    dd_loaded_lazy = DonorData(G=gdata_loaded_lazy, C=adata)
    write_variants_to_vcf(dd_loaded.G, out_file = output_path_vcf)
    write_variants_to_vcf(dd_loaded_lazy.G, out_file = output_path_vcf_lazy)
    with open(output_path_vcf, 'r') as f1, open(output_path_vcf_lazy, 'r') as f2:
        assert f1.read() == f2.read()