import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from cellink._core.data_fields import DAnn, VAnn

from ._plink import _dense_block, write_plink1

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def _obs_column(gdata: AnnData, key: str, default) -> np.ndarray:
    if key in gdata.obs.columns:
        return gdata.obs[key].to_numpy()
    return np.full(gdata.n_obs, default, dtype=object)


_SEX_LABELS = {"male": 1, "m": 1, "female": 2, "f": 2}


def _plink_sex_codes(sex: pd.Series, sex_map: dict | None = None) -> np.ndarray:
    """Translate a donor sex column to PLINK ``.fam`` codes (1 = male, 2 = female, 0 = unknown).

    Parameters
    ----------
    sex
        Donor sex annotation.
    sex_map
        Explicit mapping from the values of ``sex`` to PLINK codes; values not in the
        mapping and missing values become 0. If None, the coding is inferred:
        missing values and negative codes (-9 in PLINK, -1 from :func:`cellink.io.read_plink`)
        are unknown; text labels ``male``/``m`` and ``female``/``f`` (any case) map to 1 and 2;
        numbers that include 2 are taken as PLINK codes; numbers in {0, 1} are taken
        as 1 = male and 0 = female, the coding of :func:`cellink.resources.get_onek1k`
        (and UK Biobank field 31).

    Returns
    -------
    numpy.ndarray
        ``int32`` PLINK sex codes.
    """
    values = pd.Series(np.asarray(sex, dtype=object))
    missing = values.isna().to_numpy()
    if sex_map is not None:
        codes = values.map(lambda v: sex_map.get(v, 0) if not pd.isna(v) else 0)
        codes = pd.to_numeric(codes, errors="raise").to_numpy()
        if not np.isin(codes, [0, 1, 2]).all():
            raise ValueError("`donor_sex_map` must map to PLINK sex codes 1 (male), 2 (female) or 0 (unknown).")
        return codes.astype("int32")

    numeric = pd.to_numeric(values, errors="coerce")
    unknown = (numeric < 0).to_numpy()
    numeric = numeric.mask(unknown)
    missing = missing | unknown
    if numeric[~missing].notna().all():
        observed = set(numeric[~missing].unique())
        if not observed <= {0, 1, 2}:
            raise ValueError(
                f"Cannot interpret donor sex values {sorted(observed)} as PLINK codes; pass `donor_sex_map`, "
                "e.g. {'M': 1, 'F': 2}."
            )
        if 2 in observed:
            codes = numeric.fillna(0)
        else:
            if 0 in observed:
                logger.info(
                    "Donor sex coded 0/1 is read as 0 = female, 1 = male (as in get_onek1k) and "
                    "written as PLINK codes 2 (female) and 1 (male). Pass `donor_sex_map={0: 0, 1: 1}` if 0 means "
                    "unknown."
                )
            codes = numeric.map({0: 2, 1: 1}).fillna(0)
        return codes.to_numpy().astype("int32")

    labels = values.astype(str).str.strip().str.lower()
    codes = labels.map(_SEX_LABELS).where(~missing).fillna(0)
    return codes.to_numpy().astype("int32")


def _var_column(gdata: AnnData, key: str) -> np.ndarray:
    col = gdata.var[key]
    if isinstance(col.dtype, pd.CategoricalDtype):
        col = col.astype(str)
    return col.to_numpy()


def _sample_table(
    gdata: AnnData,
    donor_id: str,
    donor_family_id: str,
    donor_paternal_id: str,
    donor_maternal_id: str,
    donor_sex: str,
    donor_sex_map: dict | None = None,
) -> pd.DataFrame:
    """Family and sample IDs, parents and PLINK sex codes (1=male, 2=female, 0=unknown) of every donor."""
    iid = gdata.obs[donor_id].to_numpy() if donor_id in gdata.obs.columns else gdata.obs.index.to_numpy()
    fid = gdata.obs[donor_family_id].to_numpy() if donor_family_id in gdata.obs.columns else iid
    sex = (
        _plink_sex_codes(gdata.obs[donor_sex], donor_sex_map)
        if donor_sex in gdata.obs.columns
        else np.zeros(gdata.n_obs, dtype="int32")
    )
    return pd.DataFrame(
        {
            "fid": fid,
            "iid": iid,
            "father": _obs_column(gdata, donor_paternal_id, "0"),
            "mother": _obs_column(gdata, donor_maternal_id, "0"),
            "sex": sex,
        }
    )


def to_plink(
    gdata: AnnData,
    output_prefix: str | os.PathLike = None,
    donor_id: str = DAnn.donor,
    donor_family_id: str = DAnn.donor_family,
    donor_paternal_id: str = DAnn.donor_paternal_id,
    donor_maternal_id: str = DAnn.donor_maternal_id,
    donor_sex: str = DAnn.donor_sex,
    chrom: str = VAnn.chrom,
    pos: str = VAnn.pos,
    a0: str = VAnn.a0,
    a1: str = VAnn.a1,
    variant_chunk_size: int | None = None,
    donor_sex_map: dict | None = None,
) -> None:
    """
    Export genotype data from an AnnData object to PLINK1 binary format (.bed, .bim, .fam).

    Parameters
    ----------
    gdata : anndata.AnnData
        AnnData object containing genotype data in `.X` (copies of the `a1` allele: 0, 1, 2,
        or NaN for missing), with variants in `.var` and sample metadata in `.obs`. `.X` may
        be a numpy, dask or scipy sparse matrix; it is encoded in chunks of variants.
    output_prefix : str or os.PathLike
        Prefix for the output PLINK files. A trailing '.bed' is accepted and stripped.
    donor_id : str, default DAnn.donor
        Column in `gdata.obs` representing sample (individual) IDs. If the column does
        not exist, the index of `gdata.obs` is used.
    donor_family_id : str, default DAnn.donor_family
        Column in `gdata.obs` representing family IDs. If not present, `donor_id` is used instead.
    donor_paternal_id : str, default DAnn.donor_paternal_id
        Column in `gdata.obs` representing paternal IDs. If not found, all values will default to 0.
    donor_maternal_id : str, default DAnn.donor_maternal_id
        Column in `gdata.obs` representing maternal IDs. If not found, all values will default to 0.
    donor_sex : str, default "sex"
        Column in `gdata.obs` representing sample sex. It is written as PLINK codes
        1=male, 2=female, 0=unknown; see `donor_sex_map` for how values are read.
        If the column is not found, all donors are written with unknown sex (0).
    chrom : str, default VAnn.chrom
        Column in `gdata.var` representing chromosome information.
    pos : str, default VAnn.pos
        Column in `gdata.var` representing base-pair positions.
    a0 : str, default VAnn.a0
        Column in `gdata.var` representing the reference allele, written to `.bim`
        column 6 (PLINK's A2).
    a1 : str, default VAnn.a1
        Column in `gdata.var` representing the alternate allele (the allele counted by
        `.X`), written to `.bim` column 5 (PLINK's A1).
    variant_chunk_size : int, optional
        Number of variants encoded at once; by default about 16 million genotypes per chunk.
    donor_sex_map : dict, optional
        Mapping from the values of `gdata.obs[donor_sex]` to PLINK sex codes
        (1=male, 2=female, 0=unknown); unmapped and missing values are written as 0.
        If None, the coding is inferred: labels "male"/"m" and "female"/"f" (any
        case) map to 1 and 2, numeric columns containing 2 are taken as PLINK codes,
        and numeric columns with only 0 and 1 are read as 1=male, 0=female, the
        coding of :func:`cellink.resources.get_onek1k`. Pass ``{0: 0, 1: 1}`` for
        a PLINK-coded column without females.

    Returns
    -------
    None
        Writes `.bed`, `.bim`, and `.fam` files to disk using the provided prefix.

    Notes
    -----
    - Allele orientation follows PLINK 2, which reads `.bim` column 5 (A1) as ALT and
      column 6 (A2) as REF: ALT (`a1`) is written to column 5 and the `.bed` codes
      count it, so `plink2 --bfile`, tensorQTL, jaxqtl and SAIGE-QTL (``alt-first``)
      report effects for the ALT allele, and :func:`cellink.io.read_plink` returns
      the same `.X`, `a0`, `a1` and variant index. Files with REF in column 5 are
      read with :func:`cellink.io.read_plink` and ``count_a1=False``.
    - The `.bed` file is written variant-major (``cellink.io._plink``).
    - Non-integer dosages are rounded to the nearest hard call (with a warning); values
      outside [0, 2] raise a ``ValueError``.
    - The phenotype column of the `.fam` file is written as missing.
    """
    if output_prefix is None:
        raise ValueError("`output_prefix` is required.")
    output_prefix = str(output_prefix)
    if output_prefix.endswith(".bed"):
        output_prefix = output_prefix[: -len(".bed")]

    fam = _sample_table(
        gdata, donor_id, donor_family_id, donor_paternal_id, donor_maternal_id, donor_sex, donor_sex_map
    )
    fam["phenotype"] = np.full(gdata.n_obs, "-0", dtype=object)  # PLINK's missing phenotype

    # .bim column 5 (A1) = ALT, column 6 (A2) = REF, so X (copies of ALT) counts A1
    bim = pd.DataFrame(
        {
            "chrom": _var_column(gdata, chrom),
            "snp": gdata.var.index.to_numpy(),
            "cm": np.zeros(gdata.n_vars, dtype=np.float64),
            "pos": gdata.var[pos].to_numpy(),
            "a1": _var_column(gdata, a1),
            "a2": _var_column(gdata, a0),
        }
    )

    write_plink1(output_prefix, gdata.X, fam, bim, variant_chunk_size=variant_chunk_size)
    logger.info(f"Wrote PLINK 1 binary files {output_prefix}.bed/.bim/.fam ({gdata.n_obs} x {gdata.n_vars})")


def to_pgen(
    gdata: AnnData,
    output_prefix: str | os.PathLike = None,
    donor_id: str = DAnn.donor,
    donor_family_id: str = DAnn.donor_family,
    donor_paternal_id: str = DAnn.donor_paternal_id,
    donor_maternal_id: str = DAnn.donor_maternal_id,
    donor_sex: str = DAnn.donor_sex,
    chrom: str = VAnn.chrom,
    pos: str = VAnn.pos,
    a0: str = VAnn.a0,
    a1: str = VAnn.a1,
    dosages: bool | None = None,
    variant_chunk_size: int | None = None,
    donor_sex_map: dict | None = None,
) -> None:
    """
    Export genotype data from an AnnData object to PLINK 2 binary format (.pgen, .pvar, .psam).

    Parameters
    ----------
    gdata : anndata.AnnData
        AnnData object containing genotype data in `.X` (copies of the `a1` allele, from 0 to
        2, or NaN for missing), with variants in `.var` and sample metadata in `.obs`. `.X`
        may be a numpy, dask or scipy sparse matrix; it is written in chunks of variants.
    output_prefix : str or os.PathLike
        Prefix for the output files. A trailing '.pgen' is accepted and stripped.
    donor_id : str, default DAnn.donor
        Column in `gdata.obs` with the sample IDs (`.psam` IID). If the column does not
        exist, the index of `gdata.obs` is used.
    donor_family_id : str, default DAnn.donor_family
        Column in `gdata.obs` with family IDs, written as `.psam` FID. Without it, the
        `.psam` has no FID column.
    donor_paternal_id, donor_maternal_id : str
        Columns in `gdata.obs` with the parents' IDs, written as `.psam` PAT and MAT if
        either exists (missing parents as 0).
    donor_sex : str, default "sex"
        Column in `gdata.obs` with the sample sex, read as in :func:`to_plink` (see
        `donor_sex_map`) and written as 1=male, 2=female, NA=unknown.
    chrom : str, default VAnn.chrom
        Column in `gdata.var` with the chromosome.
    pos : str, default VAnn.pos
        Column in `gdata.var` with the base-pair position.
    a0 : str, default VAnn.a0
        Column in `gdata.var` with the reference allele, written as `.pvar` REF.
    a1 : str, default VAnn.a1
        Column in `gdata.var` with the alternate allele (the allele counted by `.X`),
        written as `.pvar` ALT.
    dosages : bool, optional
        True writes every variant as dosages, False writes hard calls only (non-integer
        values are rounded to the nearest call, with a warning). By default variants with
        non-integer values are written as dosages and the others as hard calls.
    variant_chunk_size : int, optional
        Number of variants written at once; by default about 16 million genotypes per chunk.
    donor_sex_map : dict, optional
        Mapping from the values of `gdata.obs[donor_sex]` to sex codes, as in :func:`to_plink`.

    Returns
    -------
    None
        Writes `.pgen`, `.pvar` and `.psam` files to disk using the provided prefix.

    Notes
    -----
    - The `.pgen` file stores copies of ALT, which is what `.X` counts, so `.X`, `a0` and
      `a1` map onto it without any allele flip.
    - PLINK 2 stores dosages in steps of 1/16384.
    - Values outside [0, 2] raise a ``ValueError``; no files are left behind.
    - Needs ``pgenlib``: ``pip install 'cellink-tools[pgen]'``.
    """
    try:
        import pgenlib
    except ImportError as e:
        raise ImportError("to_pgen requires pgenlib. Install with:\n\n    pip install 'cellink-tools[pgen]'") from e
    if output_prefix is None:
        raise ValueError("`output_prefix` is required.")
    output_prefix = str(output_prefix)
    if output_prefix.endswith(".pgen"):
        output_prefix = output_prefix[: -len(".pgen")]

    n_samples, n_variants = gdata.shape
    if variant_chunk_size is None:
        variant_chunk_size = max(1, (1 << 24) // max(n_samples, 1))
    paths = [f"{output_prefix}.{ext}" for ext in ("pgen", "pvar", "psam")]
    writer = pgenlib.PgenWriter(
        paths[0].encode(),
        n_samples,
        variant_ct=n_variants,
        nonref_flags=False,
        dosage_present=dosages is not False,
    )
    rounded = False
    try:
        for start in range(0, n_variants, variant_chunk_size):
            stop = min(start + variant_chunk_size, n_variants)
            block = np.asarray(_dense_block(gdata.X, start, stop), dtype=np.float64).T  # variants x samples
            missing = np.isnan(block)
            observed = block[~missing]
            if observed.size and (observed.min() < 0 or observed.max() > 2):
                raise ValueError(
                    f"Genotype values must lie in [0, 2] (copies of `a1`) or be NaN; variants {start}-{stop - 1} "
                    f"have values in [{observed.min()}, {observed.max()}]."
                )
            calls = np.where(missing, 0.0, np.rint(block))
            if dosages is False:
                if not rounded and not np.array_equal(calls[~missing], observed):
                    warnings.warn(
                        "Non-integer dosages are rounded to the nearest hard call (0, 1, 2); "
                        "pass dosages=None or True to keep them.",
                        UserWarning,
                        stacklevel=2,
                    )
                    rounded = True
                as_dosage = np.zeros(len(block), dtype=bool)
            elif dosages:
                as_dosage = np.ones(len(block), dtype=bool)
            else:
                as_dosage = ((calls != block) & ~missing).any(axis=1)
            # write runs of hard-call and dosage variants in variant order
            bounds = np.flatnonzero(np.diff(as_dosage.astype(np.int8))) + 1
            for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(block)], strict=True):
                run_missing = missing[lo:hi]
                if as_dosage[lo]:
                    run = np.where(run_missing, -9.0, block[lo:hi]).astype(np.float32)
                    writer.append_dosages_batch(np.ascontiguousarray(run))
                else:
                    run = np.where(run_missing, -9, calls[lo:hi]).astype(np.int8)
                    writer.append_biallelic_batch(np.ascontiguousarray(run))
        writer.close()
    except BaseException:
        try:
            writer.close()
        except Exception:  # noqa: BLE001 - the original error is the one to report
            pass
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
        raise

    pvar = pd.DataFrame(
        {
            "#CHROM": _var_column(gdata, chrom),
            "POS": np.asarray(gdata.var[pos]).astype(np.int64),
            "ID": gdata.var.index.to_numpy(),
            "REF": _var_column(gdata, a0),
            "ALT": _var_column(gdata, a1),
        }
    )
    pvar.to_csv(paths[1], sep="\t", index=False, lineterminator="\n")

    samples = _sample_table(
        gdata, donor_id, donor_family_id, donor_paternal_id, donor_maternal_id, donor_sex, donor_sex_map
    )
    psam = pd.DataFrame(index=range(n_samples))
    if donor_family_id in gdata.obs.columns:
        psam["#FID"] = samples["fid"].to_numpy()
        psam["IID"] = samples["iid"].to_numpy()
    else:
        psam["#IID"] = samples["iid"].to_numpy()
    if donor_paternal_id in gdata.obs.columns or donor_maternal_id in gdata.obs.columns:
        psam["PAT"] = samples["father"].to_numpy()
        psam["MAT"] = samples["mother"].to_numpy()
    psam["SEX"] = np.where(samples["sex"].isin([1, 2]), samples["sex"].astype(str), "NA")
    psam.to_csv(paths[2], sep="\t", index=False, lineterminator="\n")
    logger.info(f"Wrote PLINK 2 binary files {output_prefix}.pgen/.pvar/.psam ({n_samples} x {n_variants})")


def write_variants_to_vcf(gdata, out_file="variants.vcf") -> None:
    """Write unique variants from gdata to vcf file for annotation

    Parameters
    ----------
    gdata : DonorData.D
        DonorData genotype object
    out_file : str
        output file. By default "variants.vcf"
    """
    logger.info(f"number of variants to annotate: {len(gdata.var)}")
    var_df = gdata.var.reset_index()[[VAnn.chrom, VAnn.pos, VAnn.index, VAnn.a0, VAnn.a1]]
    var_df[["QUAL", "FILTER", "INFO"]] = ". . .".split()
    logger.info(f"Writing variants to {out_file}")
    with open(out_file, "w") as f:
        # Write the VCF header
        f.write("##fileformat=VCFv4.0\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        var_df.to_csv(f, sep="\t", index=False, header=False)
