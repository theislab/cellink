import mudata as md
import numpy as np
import pandas as pd
import pytest

from cellink import HiData, HiDataSchemaError, validate
from cellink._core.dummy_data import sim_adata, sim_gdata, sim_mudata
from cellink.pl import expression_by_genotype
from cellink.tl.external import calculate_ld, calculate_pcs, run_tensorqtl

if hasattr(md, "set_options"):
    md.set_options(pull_on_update=False)  # mudata <0.4
else:
    md.settings.pull_on_update = False  # mudata >=0.4 (Python >=3.12): set_options -> settings


def test_validate_mudata_group_level_checks_genotype_modality():
    adata = sim_adata()
    gdata = sim_gdata(adata=adata)
    hd = HiData(G=md.MuData({"genotype": gdata}), C=adata, group_id="donor_id")
    assert validate(hd)

    gdata_bad = gdata.copy()
    gdata_bad.var = gdata_bad.var.drop(columns=["a1"])
    with pytest.raises(HiDataSchemaError, match="genotype"):
        validate(HiData(G=md.MuData({"genotype": gdata_bad}), C=adata, group_id="donor_id"))

    gdata_none = gdata.copy()
    gdata_none.var = gdata_none.var.drop(columns=["chrom"])
    with pytest.raises(HiDataSchemaError, match="No modality"):
        validate(HiData(G=md.MuData({"genotype": gdata_none}), C=adata, group_id="donor_id"))


def _group_table_hd():
    donors = pd.DataFrame({"age": np.arange(10.0)}, index=pd.Index([f"D{i}" for i in range(10)], name="donor_id"))
    return HiData(G=donors, C=sim_adata(), group_id="donor_id")


def test_validate_checks_variants_of_the_group_level():
    adata = sim_adata()
    gdata = sim_gdata(adata=adata)
    assert validate(HiData(G=gdata, C=adata, group_id="donor_id"))
    gdata.var = gdata.var.drop(columns=["pos"])
    with pytest.raises(HiDataSchemaError, match="hd.G.var"):
        validate(HiData(G=gdata, C=adata, group_id="donor_id"))

    hd = _group_table_hd()
    assert validate(hd, check_var=False)
    with pytest.raises(HiDataSchemaError, match="group table"):
        validate(hd)


@pytest.mark.parametrize(
    "tool",
    [
        lambda hd, prefix: calculate_pcs(hd, prefix, run=False),
        lambda hd, prefix: calculate_ld(hd, prefix, run=False),
        lambda hd, prefix: run_tensorqtl(hd, mode="cis_nominal", prefix=prefix),
        lambda hd, prefix: expression_by_genotype(hd, snp="SNP0", show=False),
    ],
)
def test_genotype_tools_reject_a_group_table(tmp_path, tool):
    with pytest.raises(ValueError, match="group table"):
        tool(_group_table_hd(), str(tmp_path / "out"))


def test_sim_mudata():
    mdata = sim_mudata(n_groups=3)
    assert set(mdata.mod) == {"rna", "atac"}
    rna = mdata.mod["rna"]
    pd.testing.assert_series_equal(mdata.obs.loc[rna.obs_names, "celltype"], rna.obs["celltype"], check_dtype=False)
    assert set(mdata.obs["donor_id"].dropna()) == {"D0", "D1", "D2"}
