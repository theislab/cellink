import logging
import os

import anndata as ad
import pandas as pd

import cellink as cl
from cellink.resources._datasets_utils import plink_filter_prune, plink_kinship, preprocess_vcf_to_plink, try_liftover
from cellink.resources._utils import _download_file, _load_config, _run, get_data_home

from .._core import DonorData

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
