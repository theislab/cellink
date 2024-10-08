import warnings
from dataclasses import dataclass

import xarray as xr
import dask.dataframe as dd
import pandas as pd
import sgkit as sg
from anndata import AnnData
from anndata.utils import asarray

warnings.filterwarnings(
    "ignore",
    message="The return type of `Dataset.dims` will be changed",
    category=FutureWarning,
)


@dataclass(frozen=True)
class VAnn:
    """Variant annotation fields in GenoAnndata"""

    CHROM: str = "chrom"
    POS: str = "pos"
    A0: str = "a0"
    A1: str = "a1"
    MAF: str = "maf"


@dataclass(frozen=True)
class SgAnn:
    """Fields in SgKit Zarr Format"""

    sample: str = "samples"
    variant: str = "variants"
    contig_id: str = "contig_id"
    filter: str = "filters"
    filter_id: str = "filter_id"
    variant_allele: str = "variant_allele"


SGKIT_ZARR_TO_GDATA = {
    "variant_AF": "AF",
    "variant_ER2": "ER2",
    "variant_MAF": VAnn.MAF,
    "variant_R2": "R2",
    "variant_contig": "contig",
    "variant_id": "id",
    "variant_id_mask": "id_mask",
    "variant_position": VAnn.POS,
    "variant_quality": "quality",
}


def _get_snp_index(var: pd.DataFrame | dd.DataFrame):
    df = var[[VAnn.CHROM, VAnn.POS, VAnn.A0, VAnn.A1]].astype(str)
    index = df.apply("_".join, axis=1)
    return index


def _to_df_only_dim(gdata, dims):
    df = gdata.drop_dims(set(gdata.dims.keys()).difference({dims})).to_dataframe()
    df.index = df.index.astype(str)
    return df


def from_sgkit(sgkit_dataset: xr.Dataset) -> AnnData:
    """Read SgKit Zarr Format"""
    X = sgkit_dataset.call_genotype.data.sum(-1).T  # additive model encoding

    obs = _to_df_only_dim(sgkit_dataset, SgAnn.sample)
    var = _to_df_only_dim(sgkit_dataset, SgAnn.variant)
    var = var.rename(columns=SGKIT_ZARR_TO_GDATA)

    contig_mapping = dict(enumerate(asarray(sgkit_dataset[SgAnn.contig_id])))
    var[VAnn.CHROM] = var.contig.map(contig_mapping)
    var[[VAnn.A0, VAnn.A1]] = asarray(sgkit_dataset[SgAnn.variant_allele]).astype(str)

    first_cols = [VAnn.CHROM, VAnn.POS, VAnn.A0, VAnn.A1]
    var = var[first_cols + [c for c in var.columns if c not in first_cols]]

    index = _get_snp_index(var)
    var.index = index

    varm = {}
    if SgAnn.filter in sgkit_dataset and SgAnn.filter_id in sgkit_dataset:
        filters = pd.DataFrame(
            asarray(sgkit_dataset[SgAnn.filter]),
            columns=asarray(sgkit_dataset[SgAnn.filter_id]),
            index=var.index,
        )
        varm["filter"] = filters

    gdata = AnnData(X=X, obs=obs, var=var, varm=varm)

    return gdata
