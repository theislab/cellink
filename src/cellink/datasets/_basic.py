import hashlib
import logging
import os
import shutil
import subprocess
from os.path import expanduser, join
from pathlib import Path
from urllib.request import urlretrieve

import anndata as ad
import pandas as pd
import yaml

import cellink as cl
from cellink.datasets._utils import plink_filter_prune, plink_kinship, preprocess_vcf_to_plink, try_liftover

logging.basicConfig(level=logging.INFO)

DEFAULT_DATA_HOME = join("~", "cellink_sample_data")


def get_data_home(data_home=None):
    """Get or create the local data storage directory."""
    if data_home is None:
        data_home = os.environ.get("CELLINK_SAMPLE_DATA", DEFAULT_DATA_HOME)
    data_home = expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return Path(data_home)


def clear_data_home(data_home=None):
    """Remove all data from local cache."""
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home, ignore_errors=True)


def _sha256sum(filename):
    """Return the sha256 checksum of the file."""
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url, dest, checksum=None):
    """Download a file and verify checksum."""
    if dest.exists():
        logging.info(f"{dest} already exists")
        if checksum is None:
            logging.warning("No checksum provided, skipping verification")
            return
        logging.info("Veryifying checksum")
        if checksum and _sha256sum(dest) == checksum:
            return
        logging.info(f"{dest} exists but checksum mismatch. Re-downloading.")
        dest.unlink()

    logging.info(f"Downloading {url} to {dest}")
    urlretrieve(url, dest)
    if checksum and _sha256sum(dest) != checksum:
        raise ValueError(f"Checksum mismatch for {dest}")


def _run(cmd, cwd=None):
    """Run a system command using subprocess."""
    logging.info(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=True)
    return result


def _load_config(path):
    """Load the YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_1000genomes(config_path="./cellink/datasets/config/1000genomes.yaml", data_home=None, verify_checksum=True):
    """Main function to download and preprocess the data."""
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


def get_onek1k(config_path="./cellink/datasets/config/onek1k.yaml", data_home=None, verify_checksum=True):
    """Main function to download and preprocess the data."""
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

    dd = cl.DonorData(G=gdata, C=adata).copy()

    return dd


if __name__ == "__main__":
    get_onek1k()
    get_1000genomes()
