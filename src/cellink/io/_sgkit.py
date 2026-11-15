import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData
from anndata.utils import asarray

from cellink._core.data_fields import DAnn, VAnn

warnings.filterwarnings(
    "ignore",
    message="The return type of `Dataset.dims` will be changed",
    category=FutureWarning,
)


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


def _as_str_array(values) -> np.ndarray:
    """Cast an allele/contig array to fixed-width unicode.

    Handles bytes (``S``), object and numpy's variable-width ``StringDType`` (as zarr
    returns VCF Zarr string arrays), which is cast through ``object``.
    """
    values = np.asarray(values)
    if values.dtype.kind == "T":
        values = values.astype(object)
    return values.astype(str)


def _get_snp_index(var: pd.DataFrame) -> pd.Index:
    df = var[[VAnn.chrom, VAnn.pos, VAnn.a0, VAnn.a1]].astype(str)
    index = df.apply("_".join, axis=1)
    return pd.Index(index, name=VAnn.index)


def _to_df_only_dim(gdata, dims):
    df = gdata.drop_dims(set(gdata.sizes.keys()).difference({dims})).to_dataframe()
    df.index = df.index.astype(str)
    return df


def _collapse_multiallelic_dosage_scalar(ds_array: da.Array) -> da.Array:
    """Collapse multiallelic dosage to scalar.

    ds_array: dask array with shape either (variants, samples) (biallelic scalar DS)
              or (variants, samples, n_alt) (vector DS).
    Returns collapsed scalar DS (variants, samples) by summing over alt axis if exists.
    """
    if ds_array.ndim == 2:
        return ds_array
    return ds_array.sum(axis=-1)


def _collapse_multiallelic_ac_scalar_from_gt(gt_array: da.Array, mask_array: da.Array | None = None) -> da.Array:
    """Collapse multiallelic genotypes to scalar allele-count.

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
    hard_call: bool = True,
    keep_multiallelic: bool = False,
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
        mapping from sgkit's sample annotation keys to desired gdata.obs column
    X_field
        One of: "GT", "DS", "GP", "AC".
        - "GT": collapsed allele count (sum of non-zero allele indices) -> X (samples, variants)
        - "DS": scalar dosage (call_DS collapsed across alts) -> X
        - "GP": argmax genotype state mapped to alt-count when mapping exists -> X
        - "AC": alias for "GT"
    hard_call
        if True, returns hard calls (0,1,2); if False, returns dosage/additive encoding
    keep_multiallelic
        if True, stores extra alternate alleles beyond ALT1; default is False
    load_call_fields
        iterable of call_* keys to load as layers; default None = load all present call_ fields.
        A field with one value per call (e.g. ``call_DS``) becomes the layer ``DS``; a field with
        an extra axis (ploidy, alleles or genotypes, e.g. ``call_genotype`` or ``call_AD``) becomes
        one layer per entry of that axis (``GENOTYPE_0``, ``GENOTYPE_1``, ``AD_0``, ...). Every
        layer has the shape of ``X``, so it follows any selection of samples and variants.
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
            warnings.warn(
                "Requested X_field='GT' but call_genotype is not present. X will be unset.", UserWarning, stacklevel=2
            )
            X = None
        else:
            gt_da_data: da.Array = gt_da.data
            mask_data = mask_da.data if mask_da is not None else None
            ac_scalar = _collapse_multiallelic_ac_scalar_from_gt(gt_da_data, mask_data)
            X = ac_scalar.T

    elif X_field == "DS":
        if ds_da is None:
            warnings.warn(
                "Requested X_field='DS' but call_DS is not present. X will be unset.", UserWarning, stacklevel=2
            )
            X = None
        else:
            ds_data = ds_da.data
            ds_scalar = _collapse_multiallelic_dosage_scalar(ds_data)
            X = ds_scalar.T

    elif X_field == "GP":
        if gp_da is None:
            warnings.warn(
                "Requested X_field='GP' but call_genotype_probability not present. X unset.", UserWarning, stacklevel=2
            )
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

    if alleles_arr is not None:
        var_df[VAnn.a0] = _as_str_array(alleles_arr[:, 0])
        if alleles_arr.shape[1] > 1:
            var_df[VAnn.a1] = _as_str_array(alleles_arr[:, 1])
        if keep_multiallelic and alleles_arr.shape[1] > 2:
            for ai in range(2, alleles_arr.shape[1]):
                var_df[f"a{ai}"] = _as_str_array(alleles_arr[:, ai])

    if SgVars.contig_label in ds.data_vars:
        contigs = _as_str_array(asarray(ds[SgVars.contig_label]))
        if "variant_contig" in ds.data_vars:
            vc = asarray(ds["variant_contig"])
            try:
                var_df["chrom"] = pd.Index(vc).map(dict(enumerate(contigs)))
            except (KeyError, TypeError, ValueError):
                var_df["chrom"] = vc
            var_df["chrom"] = var_df["chrom"].astype(str)
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
        if X_field == "DS" and key == SgVars.dosage:
            continue
        if X_field == "GP" and key == SgVars.genotype_alt:
            continue

        name = key.replace("call_", "").upper()
        if da_var.data.ndim == 2:
            adata.layers[name] = da_var.data.T
        else:
            # one (samples, variants) layer per ploidy/allele/genotype entry
            for k in range(da_var.data.shape[-1]):
                adata.layers[f"{name}_{k}"] = da_var.data[:, :, k].T

    adata.uns["n_alt"] = n_alt
    if inferred_ploidy is not None:
        adata.uns["ploidy"] = inferred_ploidy

    if phased_da is not None:
        adata.uns["has_phased_flag"] = True
        phased_data = phased_da.data
        if phased_data.ndim == 3:
            ploidy = inferred_ploidy if inferred_ploidy is not None else phased_data.shape[-1]
            for h in range(ploidy):
                adata.layers[f"PHASE_{h}"] = phased_data[:, :, h].T
        else:
            # 2D (variants, samples): single phased flag per call
            adata.layers["PHASE_0"] = phased_data.T
    else:
        adata.uns["has_phased_flag"] = False

    adata.uns["has_genotype"] = gt_da is not None
    adata.uns["has_genotype_probability"] = gp_da is not None
    adata.uns["has_DS"] = ds_da is not None
    adata.uns["has_mask"] = mask_da is not None

    return adata


def read_sgkit_zarr(
    path: str | Path,
    *,
    var_rename=None,
    obs_rename=None,
    X_field: str = "GT",
    hard_call=True,
    keep_multiallelic=False,
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
        One of: "GT", "DS", "GP", "AC".
        - "GT": collapsed allele count (sum of non-zero allele indices) -> X (samples, variants)
        - "DS": scalar dosage (call_DS collapsed across alts) -> X
        - "GP": argmax genotype state mapped to alt-count when mapping exists -> X
        - "AC": alias for "GT"
    hard_call
        if True, returns hard calls (0,1,2); if False, returns dosage/additive encoding
    keep_multiallelic
        if True, stores extra alternate alleles beyond ALT1; default is False
    load_call_fields
        iterable of call_* keys to load as layers; default None = load all present call_ fields.
        A field with one value per call (e.g. ``call_DS``) becomes the layer ``DS``; a field with
        an extra axis (ploidy, alleles or genotypes, e.g. ``call_genotype`` or ``call_AD``) becomes
        one layer per entry of that axis (``GENOTYPE_0``, ``GENOTYPE_1``, ``AD_0``, ...). Every
        layer has the shape of ``X``, so it follows any selection of samples and variants.
    """
    try:
        import sgkit as sg
    except ImportError as e:
        raise ImportError(
            "sgkit is required for `read_sgkit_zarr`. Install with `pip install 'cellink-tools[datasets]'`."
        ) from e

    sgkit_dataset = sg.load_dataset(store=path, **kwargs)
    gdata = from_sgkit_dataset(
        sgkit_dataset,
        var_rename=var_rename,
        obs_rename=obs_rename,
        X_field=X_field,
        hard_call=hard_call,
        keep_multiallelic=keep_multiallelic,
        load_call_fields=load_call_fields,
    )
    return gdata


def _fam_quantitative_phenotype(fam_path, sep: str) -> np.ndarray | None:
    """Return the phenotypes of a ``.fam`` file as float if they are quantitative, else None.

    As in PLINK, a column holding only 1, 2 and the missing codes 0 and -9 is case/control;
    any other column is quantitative, with -9 and non-numeric values such as ``NA`` missing.
    """
    column = pd.read_csv(fam_path, sep=sep, header=None, usecols=[5], dtype=str, keep_default_na=False).iloc[:, 0]
    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    if np.isin(values[~np.isnan(values)], [-9, 0, 1, 2]).all():
        return None
    values[values == -9] = np.nan
    return values


def read_plink(
    path: str | Path = None,
    *,
    var_rename=None,
    obs_rename=None,
    X_field: str = "GT",
    hard_call=True,
    keep_multiallelic=False,
    load_call_fields: Iterable[str] | None = None,
    count_a1: bool = True,
    **kwargs,
) -> AnnData:
    """Read a PLINK 1 binary file set (``.bed``/``.bim``/``.fam``) with sgkit.

    With ``count_a1=True`` (default), alleles follow the PLINK 2 convention, in which
    ``.bim`` column 5 (A1) is ALT and column 6 (A2) is REF: ``var["a1"]`` is the column-5
    allele, ``var["a0"]`` the column-6 allele, and ``X`` counts copies of ``a1``. This is
    the layout written by :func:`cellink.io.to_plink` and by ``plink2 --make-bed``. Files
    from PLINK 1.9 without ``--keep-allele-order`` carry the minor allele in column 5, so
    ``a1`` is the minor allele there, not necessarily ALT.

    ``obs["sex"]`` is 1 (male), 2 (female) or -1 (unknown). ``obs["phenotype"]`` is 1
    (control), 2 (case) or -1 (missing) for a case/control ``.fam`` phenotype, and the
    values as float, with ``nan`` for -9 and non-numeric values, for a quantitative one.

    Params
    ------
    path
        Path to the file set without extension (``data`` for ``data.bed``/``.bim``/``.fam``).
        A trailing ``.bed`` is stripped.
    var_rename
        mapping from sgkit's variant annotation keys to desired gdata.var column
    obs_rename
        mapping from sgkit's sample annotation keys to desired gdata.obs column
    X_field
        One of: "GT", "DS", "GP", "AC"; see :func:`from_sgkit_dataset`.
    hard_call
        if True, returns hard calls (0,1,2); if False, returns dosage/additive encoding
    keep_multiallelic
        if True, stores extra alternate alleles beyond ALT1; default is False
    load_call_fields
        iterable of call_* keys to load as layers; default None = load all present call_ fields.
    count_a1
        If True (default), ``X`` counts the column-5 allele, stored as ``a1``. If False,
        ``X`` counts the column-6 allele, stored as ``a1``, and ``a0`` is column 5. Use
        ``False`` for files with REF in column 5 and ALT in column 6.
    kwargs
        Further keyword arguments, passed to ``sgkit.io.plink.read_plink``. ``fam_sep`` and
        ``bim_sep`` default to any whitespace, as the PLINK formats allow.
    """
    try:
        from sgkit.io import plink as sg_plink
    except ImportError as e:
        raise ImportError(
            "sgkit with its plink extra is required for `read_plink`. Install with `pip install 'cellink-tools[datasets]'`."
        ) from e

    if path is not None and str(path).endswith(".bed"):
        path = str(path)[: -len(".bed")]
    kwargs.setdefault("fam_sep", r"\s+")
    kwargs.setdefault("bim_sep", r"\s+")
    sgkit_dataset = sg_plink.read_plink(path=path, **kwargs)
    # sgkit codes every phenotype other than 1 and 2 as missing
    phenotype = _fam_quantitative_phenotype(kwargs.get("fam_path") or f"{path}.fam", kwargs["fam_sep"])
    if phenotype is not None:
        sgkit_dataset = sgkit_dataset.assign(sample_phenotype=(sgkit_dataset["sample_phenotype"].dims, phenotype))
    # sgkit stores PLINK's unknown-chromosome code 0 as a missing contig name
    contigs = [
        c if isinstance(c, str) and c not in ("None", "nan") else "0"
        for c in sgkit_dataset["contig_id"].values.tolist()
    ]
    sgkit_dataset = sgkit_dataset.assign(contig_id=(sgkit_dataset["contig_id"].dims, np.asarray(contigs, dtype=str)))
    if count_a1:
        # sgkit counts column 6; swap the alleles and the allele index of every called copy
        genotype = sgkit_dataset["call_genotype"]
        flipped = da.where(genotype.data >= 0, 1 - genotype.data, genotype.data).astype(genotype.dtype)
        sgkit_dataset = sgkit_dataset.assign(
            call_genotype=genotype.copy(data=flipped),
            variant_allele=sgkit_dataset["variant_allele"][:, ::-1],
        )
    gdata = from_sgkit_dataset(
        sgkit_dataset,
        var_rename=var_rename,
        obs_rename=obs_rename,
        X_field=X_field,
        hard_call=hard_call,
        keep_multiallelic=keep_multiallelic,
        load_call_fields=load_call_fields,
    )
    return gdata


def read_bgen(
    path: str | Path = None,
    metafile_path: str | Path = None,
    sample_path: str | Path = None,
    *,
    var_rename=None,
    obs_rename=None,
    X_field: str = "GT",
    hard_call=True,
    keep_multiallelic=False,
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
        One of: "GT", "DS", "GP", "AC".
        - "GT": collapsed allele count (sum of non-zero allele indices) -> X (samples, variants)
        - "DS": scalar dosage (call_DS collapsed across alts) -> X
        - "GP": argmax genotype state mapped to alt-count when mapping exists -> X
        - "AC": alias for "GT"
    hard_call
        if True, returns hard calls (0,1,2); if False, returns dosage/additive encoding
    keep_multiallelic
        if True, stores extra alternate alleles beyond ALT1; default is False
    load_call_fields
        iterable of call_* keys to load as layers; default None = load all present call_ fields.
        A field with one value per call (e.g. ``call_DS``) becomes the layer ``DS``; a field with
        an extra axis (ploidy, alleles or genotypes, e.g. ``call_genotype`` or ``call_AD``) becomes
        one layer per entry of that axis (``GENOTYPE_0``, ``GENOTYPE_1``, ``AD_0``, ...). Every
        layer has the shape of ``X``, so it follows any selection of samples and variants.
    """
    try:
        from sgkit.io import bgen as sg_bgen
    except ImportError as e:
        raise ImportError(
            "sgkit is required for `read_bgen`. Install with `pip install 'cellink-tools[datasets]'`."
        ) from e

    sgkit_dataset = sg_bgen.read_bgen(path=path, metafile_path=metafile_path, sample_path=sample_path, **kwargs)
    gdata = from_sgkit_dataset(
        sgkit_dataset,
        var_rename=var_rename,
        obs_rename=obs_rename,
        X_field=X_field,
        hard_call=hard_call,
        keep_multiallelic=keep_multiallelic,
        load_call_fields=load_call_fields,
    )
    return gdata
