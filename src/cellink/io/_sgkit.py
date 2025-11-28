import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
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
    dosage: str = "call_DS"
    mask: str = "call_genotype_mask"
    phased: str = "call_genotype_phased"


SGVAR_TO_GDATA = {
    "variant_MAF": VAnn.maf,
    "variant_position": VAnn.pos,
    "variant_contig": VAnn.contig,
}


# TODO
def _get_snp_index(var: pd.DataFrame) -> pd.Index:
    # Expect var to have columns named: contig/contig mapping already applied to 'chrom'
    # Use a0,a1,pos to create index "chrom_pos_a0_a1"
    parts = []
    for c in ("chrom", "pos", "a0", "a1"):
        if c not in var.columns:
            # fill with empty string
            var[c] = ""
    idx = var[["chrom", "pos", "a0", "a1"]].astype(str).apply("_".join, axis=1)
    return pd.Index(idx, name="snp_id")


def _collapse_multiallelic_dosage_scalar(ds_array: da.Array) -> da.Array:
    """
    ds_array: dask array with shape either (variants, samples) (biallelic scalar DS)
              or (variants, samples, n_alt) (vector DS).
    Returns collapsed scalar DS (variants, samples) by summing over alt axis if exists.
    """
    if ds_array.ndim == 2:
        return ds_array
    return ds_array.sum(axis=-1)


def _collapse_multiallelic_ac_scalar_from_gt(gt_array: da.Array, mask_array: da.Array | None = None) -> da.Array:
    """
    gt_array: dask array (variants, samples, ploidy)
    returns scalar allele-count of ALT (non-zero allele indices) per variant/sample
    shape -> (variants, samples)
    If mask_array provided (same shape), set entries with any masked allele to nan.
    """
    alt_count = (gt_array > 0).sum(axis=-1).astype("int16")
    if mask_array is not None:
        any_missing = mask_array.any(axis=-1)
        alt_count = alt_count.astype("float32")
        alt_count = da.where(any_missing, da.full_like(alt_count, np.nan, dtype="float32"), alt_count)
    return alt_count


def from_sgkit_dataset(
    sgkit_dataset: xr.Dataset,
    *,
    var_rename: dict | None = None,
    obs_rename: dict | None = None,
    X_field: str = "GT",
    load_call_fields: Iterable[str] | None = None,
) -> AnnData:
    """
    Convert an sgkit xarray.Dataset to AnnData.

    Parameters
    ----------
    sgkit_dataset
        xarray.Dataset from sgkit (lazy/dask-backed)
    var_rename
        mapping from sgkit variant keys (e.g., 'variant_position') to var column names
    obs_rename
        mapping for sample-level annotations
    X_field
        One of: "GT", "DS", "GP", "MASK", "AC", "NONE".
        - "GT": collapsed allele count (sum of non-zero allele indices) -> X (samples, variants)
        - "DS": scalar dosage (call_DS collapsed across alts) -> X
        - "GP": argmax genotype state mapped to alt-count when mapping exists -> X
        - "MASK": fraction of masked allele copies per (variant,sample)
        - "AC": alias for "GT"
        - "NONE": do not set X (X = np.empty((n_samples, 0))) or set to zeros? We set X to empty 2D dask array.
    load_call_fields
        iterable of call_* keys to load as layers; default None = load all present call_ fields.
    """
    var_rename = SGVAR_TO_GDATA if var_rename is None else var_rename
    obs_rename = {} if obs_rename is None else obs_rename
    load_call_fields = None if load_call_fields is None else set(load_call_fields)

    ds = sgkit_dataset

    alleles_da = ds.data_vars.get(SgVars.alleles, None)
    if alleles_da is None:
        n_alleles = 2
        alleles_arr = None
    else:
        alleles_arr = asarray(alleles_da)
        n_alleles = int(alleles_arr.shape[1])

    n_alt = max(0, n_alleles - 1)

    gt_da = ds.data_vars.get(SgVars.genotype, None)
    gp_da = ds.data_vars.get(SgVars.genotype_alt, None)
    ds_da = ds.data_vars.get(SgVars.dosage, None)
    mask_da = ds.data_vars.get(SgVars.mask, None)
    phased_da = ds.data_vars.get(SgVars.phased, None)

    inferred_ploidy = None
    if gt_da is not None:
        if gt_da.ndim != 3:
            raise ValueError(f"Unexpected genotype array ndim: {gt_da.ndim}, expected 3 (variants, samples, ploidy).")
        inferred_ploidy = int(gt_da.sizes.get("ploidy", gt_da.shape[-1]))

    X = None
    if X_field in ("GT", "AC"):
        if gt_da is None:
            warnings.warn("Requested X_field='GT' but call_genotype is not present. X will be unset.", UserWarning)
            X = None
        else:
            gt_da_data: da.Array = gt_da.data
            mask_data = mask_da.data if mask_da is not None else None
            ac_scalar = _collapse_multiallelic_ac_scalar_from_gt(gt_da_data, mask_data)
            X = ac_scalar.T

    elif X_field == "DS":
        if ds_da is None:
            warnings.warn("Requested X_field='DS' but call_DS is not present. X will be unset.", UserWarning)
            X = None
        else:
            ds_data = ds_da.data
            ds_scalar = _collapse_multiallelic_dosage_scalar(ds_data)
            X = ds_scalar.T

    elif X_field == "GP":
        if gp_da is None:
            warnings.warn("Requested X_field='GP' but call_genotype_probability not present. X unset.", UserWarning)
            X = None
        else:
            gp_data = gp_da.data
            state_idx = gp_data.argmax(axis=-1)
            if "genotype_state_allele" in gp_da.attrs:
                alleles_state = np.asarray(gp_da.attrs["genotype_state_allele"])
                alt_count_state = (alleles_state > 0).sum(axis=1)
                X = da.take(alt_count_state, state_idx).T
            else:
                X = state_idx.T
    else:
        raise ValueError(f"Unknown X_field: {X_field!r}. Must be one of GT/DS/GP/AC.")

    obs_df = _to_df_only_dim(sgkit_dataset, SgDims.samples)
    obs_df = obs_df.rename(columns=obs_rename)
    obs_df.columns = obs_df.columns.str.replace("sample_", "")
    obs_df = obs_df.set_index("id")
    obs_df.index.name = DAnn.donor

    var_df = _to_df_only_dim(sgkit_dataset, SgDims.variants)
    var_df = var_df.rename(columns=var_rename)
    var_df.columns = var_df.columns.str.replace("variant_", "")

    """
    try:
        obs_df = ds.drop_dims({SgDims.variants}).to_dataframe()
        obs_df = obs_df.rename(columns=obs_rename)
        obs_df.index = obs_df.index.astype(str)
        obs_df.index.name = "sample"
    except Exception:
        if "samples" in ds.coords:
            samples_idx = asarray(ds.coords["samples"]).astype(str)
            obs_df = pd.DataFrame(index=samples_idx)
            obs_df.index.name = "sample"
        else:
            n_samples = int(ds.sizes.get("samples", 0))
            obs_df = pd.DataFrame(index=[f"sample_{i}" for i in range(n_samples)])
            obs_df.index.name = "sample"

    try:
        var_df = ds.drop_dims({SgDims.samples}).to_dataframe()
        var_df = var_df.rename(columns=var_rename)
        var_df.index = var_df.index.astype(str)
    except Exception:
        n_variants = int(ds.sizes.get("variants", 0))
        var_df = pd.DataFrame(index=[f"var_{i}" for i in range(n_variants)])
        var_df.index.name = "variant"
    """

    if alleles_arr is not None:
        var_df["a0"] = alleles_arr[:, 0]
        if alleles_arr.shape[1] > 1:
            var_df["a1"] = alleles_arr[:, 1]
        if alleles_arr.shape[1] > 2:
            for ai in range(2, alleles_arr.shape[1]):
                var_df[f"a{ai}"] = alleles_arr[:, ai]

    if SgVars.contig_label in ds.data_vars:
        contigs = asarray(ds[SgVars.contig_label])
        if "variant_contig" in ds.data_vars:
            vc = asarray(ds["variant_contig"])
            try:
                var_df["chrom"] = pd.Index(vc).map(dict(enumerate(contigs)))
            except Exception:
                var_df["chrom"] = vc
        else:
            pass

    var_df.index = _get_snp_index(var_df)

    if X is None:
        n_samples = int(ds.sizes.get("samples", 0))
        X_for_adata = da.empty((n_samples, 0), dtype="float32")
    else:
        X_for_adata = X

    adata = AnnData(X=X_for_adata, obs=obs_df, var=var_df)

    for key, da_var in ds.data_vars.items():
        if not key.startswith("call_"):
            continue
        if load_call_fields is not None and key not in load_call_fields:
            continue
        if X_field == "GT" and key == SgVars.genotype:
            continue
        if X_field == "DS" and key == SgVars.dosage:
            continue
        if X_field == "GP" and key == SgVars.genotype_prob:
            continue

        if da_var.data.T.ndim == 2:
            adata.uns[key.replace("call_", "").upper()] = da_var.data.T
        else:
            adata.layers[key.replace("call_", "").upper()] = da_var.data.T

    adata.uns["n_alt"] = n_alt
    if inferred_ploidy is not None:
        adata.uns["ploidy"] = inferred_ploidy

    if phased_da is not None:
        adata.uns["has_phased_flag"] = True
        adata.layers["PHASED_FLAG"] = phased_da.data.T
    else:
        adata.uns["has_phased_flag"] = False

    adata.uns["has_genotype"] = gt_da is not None
    adata.uns["has_genotype_probability"] = gp_da is not None
    adata.uns["has_DS"] = ds_da is not None
    adata.uns["has_mask"] = mask_da is not None

    if gt_da is not None:
        adata.uns["raw_call_genotype"] = gt_da.data
    if gp_da is not None:
        adata.uns["raw_call_genotype_probability"] = gp_da.data
    if ds_da is not None:
        adata.uns["raw_call_DS"] = ds_da.data
    if mask_da is not None:
        adata.uns["raw_call_genotype_mask"] = mask_da.data

    return adata


def read_sgkit_zarr(
    path: str | Path,
    *,
    var_rename=None,
    obs_rename=None,
    X_field: str = "GT",
    load_call_fields: Iterable[str] | None = None,
    **kwargs,
) -> AnnData:
    """Read SgKit Zarr Format

    Params
    ------
    path
        path to sgkit zarr format
    var_rename
        mapping from sgkit's variant annotation keys to desired gdata.var column
    obs_rename
        mapping from sgkit's sample annotation keys to desired gdata.obs column
    X_field
        One of: "GT", "DS", "GP", "MASK", "AC", "NONE".
        - "GT": collapsed allele count (sum of non-zero allele indices) -> X (samples, variants)
        - "DS": scalar dosage (call_DS collapsed across alts) -> X
        - "GP": argmax genotype state mapped to alt-count when mapping exists -> X
        - "MASK": fraction of masked allele copies per (variant,sample)
        - "AC": alias for "GT"
        - "NONE": do not set X (X = np.empty((n_samples, 0))) or set to zeros? We set X to empty 2D dask array.
    load_call_fields
        iterable of call_* keys to load as layers; default None = load all present call_ fields.
    """
    sgkit_dataset = sg.load_dataset(store=path, **kwargs)
    gdata = from_sgkit_dataset(
        sgkit_dataset, var_rename=var_rename, obs_rename=obs_rename, X_field=X_field, load_call_fields=load_call_fields
    )
    return gdata


def read_plink(
    path: str | Path = None,
    *,
    var_rename=None,
    obs_rename=None,
    X_field: str = "GT",
    load_call_fields: Iterable[str] | None = None,
    **kwargs,
) -> AnnData:
    """Read Plink Format

    Params
    ------
    path
        path to plink format
    var_rename
        mapping from sgkit's variant annotation keys to desired gdata.var column
    obs_rename
        mapping from sgkit's sample annotation keys to desired gdata.obs column
    X_field
        One of: "GT", "DS", "GP", "MASK", "AC", "NONE".
        - "GT": collapsed allele count (sum of non-zero allele indices) -> X (samples, variants)
        - "DS": scalar dosage (call_DS collapsed across alts) -> X
        - "GP": argmax genotype state mapped to alt-count when mapping exists -> X
        - "MASK": fraction of masked allele copies per (variant,sample)
        - "AC": alias for "GT"
        - "NONE": do not set X (X = np.empty((n_samples, 0))) or set to zeros? We set X to empty 2D dask array.
    load_call_fields
        iterable of call_* keys to load as layers; default None = load all present call_ fields.
    """
    from sgkit.io import plink as sg_plink

    sgkit_dataset = sg_plink.read_plink(path=path, **kwargs)
    gdata = from_sgkit_dataset(
        sgkit_dataset, var_rename=var_rename, obs_rename=obs_rename, X_field=X_field, load_call_fields=load_call_fields
    )
    return gdata


def read_bgen(
    path: str | Path = None,
    *,
    var_rename=None,
    obs_rename=None,
    X_field: str = "GT",
    load_call_fields: Iterable[str] | None = None,
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
    X_field
        One of: "GT", "DS", "GP", "MASK", "AC", "NONE".
        - "GT": collapsed allele count (sum of non-zero allele indices) -> X (samples, variants)
        - "DS": scalar dosage (call_DS collapsed across alts) -> X
        - "GP": argmax genotype state mapped to alt-count when mapping exists -> X
        - "MASK": fraction of masked allele copies per (variant,sample)
        - "AC": alias for "GT"
        - "NONE": do not set X (X = np.empty((n_samples, 0))) or set to zeros? We set X to empty 2D dask array.
    load_call_fields
        iterable of call_* keys to load as layers; default None = load all present call_ fields.
    """
    from sgkit.io import bgen as sg_bgen

    sgkit_dataset = sg_bgen.read_bgen(path=path, **kwargs)
    gdata = from_sgkit_dataset(
        sgkit_dataset, var_rename=var_rename, obs_rename=obs_rename, X_field=X_field, load_call_fields=load_call_fields
    )
    return gdata
