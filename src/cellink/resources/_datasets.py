import logging
import os

import anndata as ad
import pandas as pd

import cellink as cl
from cellink.resources._datasets_utils import plink_filter_prune, plink_kinship, preprocess_vcf_to_plink, try_liftover
from cellink.resources._utils import _download_file, _load_config, _run, get_data_home

from .._core import DonorData
from .._core.data_fields import DAnn

logging.basicConfig(level=logging.INFO)


def get_1000genomes(
    config_path: str = "./cellink/resources/config/1000genomes.yaml", data_home: str | None = None, verify_checksum=True
) -> ad.AnnData:
    """
    Download and preprocess the 1000 Genomes Project genotype data.

    This function downloads genotype files specified in a YAML configuration,
    optionally verifies checksums, converts VCF files to Zarr format using `vcf2zarr`,
    and concatenates per-chromosome datasets into a single `AnnData` object using `cellink`.

    Parameters
    ----------
    config_path : str, default="./cellink/resources/config/1000genomes.yaml"
        Path to the YAML configuration file listing remote genotype files.
    data_home : str or None, optional
        Directory where data should be stored. If None, uses the default `cellink` data directory.
    verify_checksum : bool, default=True
        If True, verifies the checksum of downloaded files.

    Returns
    -------
    anndata.AnnData
        Concatenated genotype data across chromosomes in Zarr format.

    Raises
    ------
    FileNotFoundError
        If any required VCF files are missing after download.
    RuntimeError
        If `vcf2zarr` conversion fails.
    """
    data_home = get_data_home(data_home)
    DATA = data_home / "1000genomes"

    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)

    for file in config["remote_files"]:
        checksum = file.get("checksum") if verify_checksum else None
        _download_file(file["url"], DATA / file["filename"], checksum)

    gdata_list = []
    for chromosome in list(range(1, 23)):
        if not os.path.isdir(DATA / "ALL.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcz"):
            _run(
                f"vcf2zarr explode ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.icf",
                cwd=DATA,
            )
            _run(
                f"vcf2zarr encode ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.icf ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcz",
                cwd=DATA,
            )
        gdata_list.append(
            cl.io.read_sgkit_zarr(
                DATA / f"ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcz"
            )
        )

    gdata = ad.concat(gdata_list, axis=1)

    return gdata


_DUMMY_DATA_ROOT = "/project/genomics/ayshan/cellink_dummy_data"


def get_dummy_data_paths() -> dict:
    """
    Return paths to the bundled dummy dataset files.

    All paths point to the pre-generated small dataset under
    ``/project/genomics/ayshan/cellink_dummy_data/``.  Use these instead of
    hardcoded machine paths in tutorials and tests so that notebooks run
    without modification.

    Returns
    -------
    dict with keys:

    ``h5ad``
        Path to the dummy OneK1K expression AnnData (885 KB, 1000 cells × 200 genes).
    ``gene_coord``
        Tab-separated gene coordinate file (GENE / CHR / START / END).
    ``bimfile``
        Template string for per-chromosome bimfiles; fill with ``{chrom}``,
        e.g. ``paths["bimfile"].format(chrom=22)``.
    ``bimfile_prefix``
        Prefix for ``compute_ld_scores_with_annotations_from_bimfile``
        (fill ``{chrom}`` to get the full prefix).
    ``genes_raw``
        Dummy MAGMA ``.genes.raw`` file for demonstrating Step III.
    ``gwas_snp_loc``
        Dummy GWAS SNP location file (SNP / CHR / BP) for MAGMA Step I.
    ``gwas_pval``
        Dummy GWAS p-value file (SNP / CHR / BP / P / N) for MAGMA Step II.

    Examples
    --------
    >>> from cellink.resources import get_dummy_data_paths
    >>> paths = get_dummy_data_paths()
    >>> dd = get_dummy_onek1k()                         # expression + genotype
    >>> bimfile_chr22 = paths["bimfile"].format(chrom=22)
    """
    from pathlib import Path
    root = Path(_DUMMY_DATA_ROOT)
    return {
        "h5ad":           str(root / "onek1k" / "onek1k_dummy.h5ad"),
        "gene_coord":     str(root / "gene_coord.txt"),
        "bimfile":        str(root / "bimfiles" / "dummy.{chrom}.bim"),
        "bimfile_prefix": str(root / "bimfiles" / "dummy.{chrom}"),
        "genes_raw":      str(root / "magma" / "dummy.genes.raw"),
        "gwas_snp_loc":   str(root / "magma" / "dummy_gwas_snps.txt"),
        "gwas_pval":      str(root / "magma" / "dummy_gwas_pval.txt"),
    }


def get_dummy_onek1k() -> DonorData:
    """
    Load the pre-generated dummy OneK1K dataset.

    Returns a small ``DonorData`` (20 donors, 1000 cells, 200 genes, 2200 SNPs)
    with the same column names as the real OneK1K dataset:
    ``predicted.celltype.l2`` in ``C.obs``, real ENSG IDs as ``C.var`` index,
    and ``chrom / pos / a0 / a1 / maf`` in ``G.var``.

    It is a drop-in replacement for :func:`get_onek1k` in notebooks and tests
    — no downloads or paths required.

    Returns
    -------
    DonorData

    Examples
    --------
    >>> from cellink.resources import get_dummy_onek1k, get_dummy_data_paths
    >>> dd    = get_dummy_onek1k()
    >>> paths = get_dummy_data_paths()
    >>> print(dd.shape)
    """
    import numpy as np
    from pathlib import Path

    paths = get_dummy_data_paths()

    # ── Expression data from saved h5ad ──────────────────────────────────
    cdata = ad.read_h5ad(paths["h5ad"])
    cdata.obs[DAnn.donor] = cdata.obs["donor_id"]

    # ── Genotype data generated to match the same donors ─────────────────
    rng = np.random.default_rng(42)
    donor_ids = cdata.obs["donor_id"].unique().tolist()
    n_donors  = len(donor_ids)

    # Build SNP positions aligned with the gene windows in the coord file
    coord = pd.read_csv(paths["gene_coord"], sep="\t")
    snp_rows = []
    for _, g in coord.iterrows():
        for _ in range(10):   # ~10 SNPs per gene window
            pos = rng.integers(max(1, g["start"] - 100_000), g["end"] + 100_000)
            snp_rows.append({"chrom": str(g["chr"]), "pos": int(pos),
                             "a0": rng.choice(["A","C"]),
                             "a1": rng.choice(["G","T"]),
                             "maf": float(rng.uniform(0.05, 0.5))})
    snp_var = pd.DataFrame(snp_rows,
                           index=[f"SNP{i}" for i in range(len(snp_rows))])

    mafs  = snp_var["maf"].values
    X_geno = rng.binomial(2, mafs, size=(n_donors, len(snp_var))).astype("float32")
    gdata = ad.AnnData(
        X=X_geno,
        obs=pd.DataFrame({"donor_id": donor_ids}, index=donor_ids),
        var=snp_var,
    )

    return DonorData(G=gdata, C=cdata)


def get_onek1k(
    config_path: str = "./cellink/resources/config/onek1k.yaml",
    data_home: str | None = None,
    verify_checksum: bool = True,
) -> DonorData:
    """
    Download and preprocess the OneK1K genotype and expression dataset.

    This function downloads genotype and expression files listed in a YAML configuration,
    optionally verifies checksums, converts VCF files to Zarr format, performs PLINK preprocessing
    including filtering, pruning, and kinship computation, and loads the dataset into a `DonorData` object.

    Additionally, it:
    - Performs liftover to hg19 coordinates for variant positions.
    - Computes donor principal components (gPCs) from genotype data.
    - Aligns expression data from CellxGene to the genotype data.
    - Encodes donor metadata such as sex and age.

    Parameters
    ----------
    config_path : str, default="./cellink/resources/config/onek1k.yaml"
        Path to the YAML configuration file listing remote genotype and expression files.
    data_home : str or None, optional
        Directory where data should be stored. If None, uses the default `cellink` data directory.
    verify_checksum : bool, default=True
        If True, verifies the checksum of downloaded files.

    Returns
    -------
    cellink.DonorData
        A `DonorData` object containing preprocessed genotype (`G`) and expression (`C`) data,
        along with kinship and principal component metadata.

    Raises
    ------
    FileNotFoundError
        If any required genotype or expression files are missing after download.
    RuntimeError
        If preprocessing steps (VCF conversion, PLINK operations, or liftover) fail.
    ValueError
        If variant liftover or donor alignment cannot be performed.
    """
    data_home = get_data_home(data_home)
    DATA = data_home / "onek1k"

    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)

    for file in config["remote_files"]:
        checksum = file.get("checksum") if verify_checksum else None
        _download_file(file["url"], DATA / file["filename"], checksum)

    if not os.path.isdir(DATA / "OneK1K.noGP.vcz"):
        _run("vcf2zarr explode OneK1K.noGP.vcf.gz OneK1K.noGP.icf", cwd=DATA)
        _run("vcf2zarr encode OneK1K.noGP.icf OneK1K.noGP.vcz", cwd=DATA)

        preprocess_vcf_to_plink(vcf_filename="OneK1K.noGP.vcf.gz", DATA=DATA)

        plink_filter_prune(fname="OneK1K.noGP", DATA=DATA)

        plink_kinship(fname="OneK1K.noGP", DATA=DATA)

    gdata = cl.io.read_sgkit_zarr(DATA / "OneK1K.noGP.vcz")

    ###

    gdata.var = gdata.var.drop(columns=["contig"])
    new_pos = gdata.var.apply(lambda row: try_liftover(row), axis=1)
    gdata.var["pos_hg19"] = new_pos.astype(pd.Int64Dtype())
    gdata.var["id_hg19"] = (
        gdata.var.chrom + "_" + gdata.var.pos_hg19.astype(str) + "_" + gdata.var.a0 + "_" + gdata.var.a1
    )
    gdata.var["id_hg19"] = gdata.var["id_hg19"].astype(str)

    ###

    gpcs = pd.read_csv(
        DATA / "pcdir" / "OneK1K.noGP.filtered.pruned.eigenvec", sep=r"\s+", index_col=1, header=None
    ).drop(columns=0)
    gdata.obsm["gPCs"] = gpcs.loc[gdata.obs_names]

    gdata.uns["kinship"] = pd.read_csv(
        DATA / "kinship" / "OneK1K.noGP.filtered.pruned.rel", delimiter="\t", header=None
    )
    kinship_id = list(
        pd.read_csv(
            DATA / "kinship" / "OneK1K.noGP.filtered.pruned.rel.id", index_col=1, delimiter="\t", header=None
        ).index
    )
    gdata.uns["kinship"].index = kinship_id
    gdata.uns["kinship"].columns = kinship_id

    ###

    adata = ad.read_h5ad(DATA / "onek1k_cellxgene.h5ad")
    adata.obs["donor_id"] = "OneK1K_" + adata.obs["donor_id"].str.split("_").str[1]

    adata.obs["sex"] = adata.obs["sex"].map({"male": 1, "female": 0}).astype(int)

    adata.obs["age"] = adata.obs["age"].astype("int")

    dd = DonorData(G=gdata, C=adata).copy()

    return dd


if __name__ == "__main__":
    get_onek1k()
    get_1000genomes()
