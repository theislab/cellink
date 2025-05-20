import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import sgkit as sg
import xarray as xr
from anndata import AnnData
from anndata.utils import asarray
from sgkit.io import plink as sg_plink

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


SGVAR_TO_GDATA = {
    "variant_MAF": VAnn.maf,
    "variant_position": VAnn.pos,
    "variant_contig": VAnn.contig,
}


def from_sgkit_dataset(sgkit_dataset: xr.Dataset, *, var_rename: dict = None, obs_rename: dict = None) -> AnnData:
    """Read SgKit Zarr Format

    Params
    ------
    sgkit_dataset
        sgkit's xarray datastructure
    var_rename
        mapping from sgkit's variant annotation keys to desired gdata.var column
    obs_rename
        mapping from sgkit's sample annotation keys to desired gdata.obs column
    """
    var_rename = SGVAR_TO_GDATA if var_rename is None else var_rename
    obs_rename = {} if obs_rename is None else obs_rename

    X = sgkit_dataset[SgVars.genotype].data.sum(-1).T  # additive model encoding

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

    alleles = asarray(sgkit_dataset[SgVars.alleles]).astype(str)

    a0_a1 = alleles[:, :2]
    var[[VAnn.a0, VAnn.a1]] = a0_a1
    if alleles.shape[1] > 2:
        var[[f"{VAnn.asymb}{i}" for i in range(alleles[:, 2:].shape[1])]] = alleles[:, 2:]
    # var[[VAnn.a0, VAnn.a1]] = asarray(sgkit_dataset[SgVars.alleles]).astype(str)

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


def read_sgkit_zarr(path: str | Path, *, var_rename=None, obs_rename=None, **kwargs) -> AnnData:
    """Read SgKit Zarr Format

    Params
    ------
    path
        path to sgkit zarr format
    var_rename
        mapping from sgkit's variant annotation keys to desired gdata.var column
    obs_rename
        mapping from sgkit's sample annotation keys to desired gdata.obs column
    """
    sgkit_dataset = sg.load_dataset(store=path, **kwargs)
    gdata = from_sgkit_dataset(sgkit_dataset, var_rename=var_rename, obs_rename=obs_rename)
    return gdata


def read_plink(path: str | Path = None, *, var_rename=None, obs_rename=None, **kwargs) -> AnnData:
    """Read Plink Format

    Params
    ------
    path
        path to plink format
    var_rename
        mapping from sgkit's variant annotation keys to desired gdata.var column
    obs_rename
        mapping from sgkit's sample annotation keys to desired gdata.obs column
    """
    sgkit_dataset = sg_plink.read_plink(path=path, **kwargs)
    gdata = from_sgkit_dataset(sgkit_dataset, var_rename=var_rename, obs_rename=obs_rename)
    return gdata
