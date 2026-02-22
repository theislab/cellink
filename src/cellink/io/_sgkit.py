import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import sgkit as sg
import xarray as xr
from anndata import AnnData
from anndata.utils import asarray

from cellink._core.data_fields import DAnn, VAnn

warnings.filterwarnings(
    "ignore",
    message="The return type of `Dataset.dims` will be changed",
    category=FutureWarning,
)


def _get_snp_index(var: pd.DataFrame) -> pd.Index:
    df = var[[VAnn.chrom, VAnn.pos, VAnn.a0, VAnn.a1]].astype(str)
    index = df.apply("_".join, axis=1)
    return pd.Index(index, name=VAnn.index)


def _to_df_only_dim(gdata, dims):
    df = gdata.drop_dims(set(gdata.sizes.keys()).difference({dims})).to_dataframe()
    df.index = df.index.astype(str)
    return df


@dataclass(frozen=True)
class SgDims:
    """Dims in SgKit Zarr Format"""

    samples: str = "samples"
    variants: str = "variants"


@dataclass(frozen=True)
class SgVars:
    """Vars in SgKit Zarr Format"""

    contig_label: str = "contig_id"  # maps the variant_contig column to chromosome name
    alleles: str = "variant_allele"
    filter: str = "variant_filter"  # idx for vcf genotype filters
    filter_id: str = "filter_id"  # maps variant_filter to filter name
    genotype: str = "call_genotype"
    genotype_alt: str = "call_genotype_probability"


SGVAR_TO_GDATA = {
    "variant_MAF": VAnn.maf,
    "variant_position": VAnn.pos,
    "variant_contig": VAnn.contig,
}


def from_sgkit_dataset(
    sgkit_dataset: xr.Dataset, 
    *, 
    var_rename: dict = None, 
    obs_rename: dict = None, 
    hard_call: bool = True, 
    keep_multiallelic: bool = False,
) -> AnnData:
    """Read SgKit Zarr Format

    Params
    ------
    sgkit_dataset
        sgkit's xarray datastructure
    var_rename
        mapping from sgkit's variant annotation keys to desired gdata.var column
    obs_rename
        mapping from sgkit's sample annotation keys to desired gdata.obs column
    hard_call
        if True, returns hard calls (0,1,2); if False, returns dosage/additive encoding
    keep_multiallelic
        if True, stores extra alternate alleles beyond ALT1; default is False
    """
    var_rename = SGVAR_TO_GDATA if var_rename is None else var_rename
    obs_rename = {} if obs_rename is None else obs_rename

    if SgVars.genotype in sgkit_dataset:
        X = sgkit_dataset[SgVars.genotype].data.sum(-1).T
    elif SgVars.genotype_alt in sgkit_dataset:
        prob = sgkit_dataset[SgVars.genotype_alt].data
        if hard_call:
            # hard-call
            X = prob.argmax(-1).T
        else:
            # dosage/additive encoding
            X = (prob[..., 1] + 2 * prob[..., 2]).T
    else:
        raise KeyError("No genotype or genotype_probability found in dataset.")

    obs = _to_df_only_dim(sgkit_dataset, SgDims.samples)
    obs = obs.rename(columns=obs_rename)
    obs.columns = obs.columns.str.replace("sample_", "")
    obs = obs.set_index("id")
    obs.index.name = DAnn.donor

    var = _to_df_only_dim(sgkit_dataset, SgDims.variants)
    var = var.rename(columns=var_rename)
    var.columns = var.columns.str.replace("variant_", "")

    contig_mapping = dict(enumerate(asarray(sgkit_dataset[SgVars.contig_label])))
    var[VAnn.chrom] = var[VAnn.contig].map(contig_mapping)

    """
    alleles = asarray(sgkit_dataset[SgVars.alleles]).astype(str)

    a0_a1 = alleles[:, :2]
    var[[VAnn.a0, VAnn.a1]] = a0_a1
    if alleles.shape[1] > 2:
        var[[f"{VAnn.asymb}{i}" for i in range(alleles[:, 2:].shape[1])]] = alleles[:, 2:]
    """
    # var[[VAnn.a0, VAnn.a1]] = asarray(sgkit_dataset[SgVars.alleles]).astype(str)
    alleles = sgkit_dataset[SgVars.alleles]

    a0 = alleles.isel(alleles=0).astype(str).compute()
    a1 = alleles.isel(alleles=1).astype(str).compute()
    var[VAnn.a0] = a0.values
    var[VAnn.a1] = a1.values

    if keep_multiallelic and alleles.sizes["alleles"] > 2:
        for i in range(2, alleles.sizes["alleles"]):
            col = alleles.isel(alleles=i).astype(str).compute()
            var[f"{VAnn.asymb}{i-2}"] = col.values

    first_cols = [VAnn.chrom, VAnn.pos, VAnn.a0, VAnn.a1]
    var = var[first_cols + [c for c in var.columns if c not in first_cols]]

    var.index = _get_snp_index(var)

    varm = {}
    if SgVars.filter in sgkit_dataset and SgVars.filter_id in sgkit_dataset:
        filters = pd.DataFrame(
            asarray(sgkit_dataset[SgVars.filter]),
            columns=asarray(sgkit_dataset[SgVars.filter_id]),
            index=var.index,
        )
        varm["filter"] = filters

    gdata = AnnData(X=X, obs=obs, var=var, varm=varm)

    return gdata


def read_sgkit_zarr(path: str | Path, *, var_rename=None, obs_rename=None, hard_call=True, keep_multiallelic=False, **kwargs) -> AnnData:
    """Read SgKit Zarr Format

    Params
    ------
    path
        path to sgkit zarr format
    var_rename
        mapping from sgkit's variant annotation keys to desired gdata.var column
    obs_rename
        mapping from sgkit's sample annotation keys to desired gdata.obs column
    hard_call
        if True, returns hard calls (0,1,2); if False, returns dosage/additive encoding
    """
    sgkit_dataset = sg.load_dataset(store=path, **kwargs)
    gdata = from_sgkit_dataset(sgkit_dataset, var_rename=var_rename, obs_rename=obs_rename, hard_call=hard_call, keep_multiallelic=keep_multiallelic)
    return gdata


def read_plink(path: str | Path = None, *, var_rename=None, obs_rename=None, hard_call=True, keep_multiallelic=False, **kwargs) -> AnnData:
    """Read Plink Format

    Params
    ------
    path
        path to plink format
    var_rename
        mapping from sgkit's variant annotation keys to desired gdata.var column
    obs_rename
        mapping from sgkit's sample annotation keys to desired gdata.obs column
    hard_call
        if True, returns hard calls (0,1,2); if False, returns dosage/additive encoding
    """
    from sgkit.io import plink as sg_plink

    sgkit_dataset = sg_plink.read_plink(path=path, **kwargs)
    gdata = from_sgkit_dataset(sgkit_dataset, var_rename=var_rename, obs_rename=obs_rename, hard_call=hard_call, keep_multiallelic=keep_multiallelic)
    return gdata


def read_bgen(
    path: str | Path = None,
    metafile_path: str | Path = None,
    sample_path: str | Path = None,
    *,
    var_rename=None,
    obs_rename=None,
    hard_call=True,
    keep_multiallelic=False,
    **kwargs,
) -> AnnData:
    """Read bgen Format

    Params
    ------
    path
        path to bgen format
    var_rename
        mapping from sgkit's variant annotation keys to desired gdata.var column
    obs_rename
        mapping from sgkit's sample annotation keys to desired gdata.obs column
    hard_call
        if True, returns hard calls (0,1,2); if False, returns dosage/additive encoding
    """
    from sgkit.io import bgen as sg_bgen

    sgkit_dataset = sg_bgen.read_bgen(path=path, metafile_path=metafile_path, sample_path=sample_path, **kwargs)
    gdata = from_sgkit_dataset(sgkit_dataset, var_rename=var_rename, obs_rename=obs_rename, hard_call=hard_call, keep_multiallelic=keep_multiallelic)

    return gdata
