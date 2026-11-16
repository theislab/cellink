import mudata as md
import numpy as np
import pandas as pd
import pytest

from cellink import DonorData, DonorDataSchemaError, validate
from cellink._core.dummy_data import sim_adata, sim_gdata, sim_mudata
from cellink.pl import expression_by_genotype
from cellink.tl.external import calculate_ld, calculate_pcs, run_tensorqtl

if hasattr(md, "set_options"):
    md.set_options(pull_on_update=False)  # mudata <0.4
else:
    md.settings.pull_on_update = False  # mudata >=0.4 (Python >=3.12): set_options -> settings


def test_validate_mudata_donor_level_checks_genotype_modality():
    adata = sim_adata()
    gdata = sim_gdata(adata=adata)
    dd = DonorData(D=md.MuData({"genotype": gdata}), C=adata)
    assert validate(dd)

    gdata_bad = gdata.copy()
    gdata_bad.var = gdata_bad.var.drop(columns=["a1"])
    with pytest.raises(DonorDataSchemaError, match="genotype"):
        validate(DonorData(D=md.MuData({"genotype": gdata_bad}), C=adata))

    gdata_none = gdata.copy()
    gdata_none.var = gdata_none.var.drop(columns=["chrom"])
    with pytest.raises(DonorDataSchemaError, match="No modality"):
        validate(DonorData(D=md.MuData({"genotype": gdata_none}), C=adata))


def _donor_table_dd():
    donors = pd.DataFrame({"age": np.arange(10.0)}, index=pd.Index([f"D{i}" for i in range(10)], name="donor_id"))
    return DonorData(D=donors, C=sim_adata())


def test_validate_checks_variants_of_the_donor_level():
    adata = sim_adata()
    gdata = sim_gdata(adata=adata)
    assert validate(DonorData(D=gdata, C=adata))
    gdata.var = gdata.var.drop(columns=["pos"])
    with pytest.raises(DonorDataSchemaError, match="dd.D.var"):
        validate(DonorData(D=gdata, C=adata))

    dd = _donor_table_dd()
    assert validate(dd, check_var=False)
    with pytest.raises(DonorDataSchemaError, match="donor table"):
        validate(dd)


@pytest.mark.parametrize(
    "tool",
    [
        lambda dd, prefix: calculate_pcs(dd, prefix, run=False),
        lambda dd, prefix: calculate_ld(dd, prefix, run=False),
        lambda dd, prefix: run_tensorqtl(dd, mode="cis_nominal", prefix=prefix),
        lambda dd, prefix: expression_by_genotype(dd, snp="SNP0", show=False),
    ],
)
def test_genotype_tools_reject_a_donor_table(tmp_path, tool):
    with pytest.raises(ValueError, match="donor table"):
        tool(_donor_table_dd(), str(tmp_path / "out"))


def test_sim_mudata():
    mdata = sim_mudata(n_donors=3)
    assert set(mdata.mod) == {"rna", "atac"}
    rna = mdata.mod["rna"]
    pd.testing.assert_series_equal(mdata.obs.loc[rna.obs_names, "celltype"], rna.obs["celltype"], check_dtype=False)
    assert set(mdata.obs["donor_id"].dropna()) == {"D0", "D1", "D2"}
