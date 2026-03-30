import logging
import os
import shutil

import anndata as ad
import pandas as pd

import cellink as cl
from cellink._core import DonorData
from cellink.io import read_h5_dd, read_zarr_dd
from cellink.resources._datasets_utils import plink_filter_prune, plink_kinship, preprocess_vcf_to_plink, try_liftover
from cellink.resources._utils import _download_file, _load_config, _run, get_data_home

logging.basicConfig(level=logging.INFO)


from pathlib import Path
from typing import Iterable

def _get_1000genomes_base(
    *,
    dataset_name: str,
    config_path: str,
    data_home: str | None,
    verify_checksum: bool,
    only_download: bool,
    rerun_preprocessing: bool,
    worker_processes: int,
    max_memory: str | None,
    variants_chunk_size: int,
    samples_chunk_size: int,
    vcf_pattern: str,
    icf_name: str,
    vcz_name: str,
    chromosomes: Iterable[int] = range(1, 23),
) -> ad.AnnData | None:
    
    data_home = get_data_home(data_home)
    DATA = data_home / dataset_name
    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)

    for file in config["remote_files"]:
        checksum = file.get("checksum") if verify_checksum else None
        _download_file(file["url"], DATA / file["filename"], checksum)

    if only_download:
        return None

    vcz_path = DATA / vcz_name
    icf_path = DATA / icf_name

    if not vcz_path.is_dir() or rerun_preprocessing:

        vcf_names = [
            vcf_pattern.format(chrom=chrom)
            for chrom in chromosomes
        ]

        for vcf_name in vcf_names:
            if not (DATA / vcf_name).exists():
                raise FileNotFoundError(f"VCF file not found: {DATA / vcf_name}")

        vcf_files_str = " ".join(vcf_names)

        try:
            _run(
                f"vcf2zarr explode -p {worker_processes} {vcf_files_str} {icf_path.name}",
                cwd=DATA,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"vcf2zarr explode failed for {dataset_name}. "
                f"Original error: {e}"
            ) from e

        encode_cmd = (
            f"vcf2zarr encode"
            f" -p {worker_processes}"
            f" -l {variants_chunk_size}"
            f" -w {samples_chunk_size}"
        )

        if worker_processes > 0 and max_memory is not None:
            encode_cmd += f" -M {max_memory}"

        encode_cmd += f" {icf_path.name} {vcz_path.name}"

        try:
            _run(encode_cmd, cwd=DATA)
        except RuntimeError as e:
            raise RuntimeError(
                f"vcf2zarr encode failed for {dataset_name}. "
                f"(variants_chunk_size={variants_chunk_size}, "
                f"samples_chunk_size={samples_chunk_size}, "
                f"max_memory={max_memory}) "
                f"Original error: {e}"
            ) from e

    return cl.io.read_sgkit_zarr(vcz_path)

def get_1000genomes(
    config_path: str = "./cellink/resources/config/1000genomes.yaml",
    data_home: str | None = None,
    verify_checksum: bool = True,
    only_download: bool = False,
    rerun_preprocessing: bool = False,
    worker_processes: int = 0,
    max_memory: str | None = None,
    variants_chunk_size: int = 32000,
    samples_chunk_size: int = 2504,
) -> ad.AnnData | None:
    """
    Download and preprocess the 1000 Genomes Project genotype data.

    This function downloads genotype files specified in a YAML configuration,
    optionally verifies checksums, converts all VCF files across chromosomes to a
    single Zarr store using `vcf2zarr`, and loads the result into an `AnnData` object
    using `cellink`.

    Parameters
    ----------
    config_path : str, default="./cellink/resources/config/1000genomes.yaml"
        Path to the YAML configuration file listing remote genotype files.
    data_home : str or None, optional
        Directory where data should be stored. If None, uses the default `cellink` data directory.
    verify_checksum : bool, default=True
        If True, verifies the checksum of downloaded files.
    only_download : bool, default=False
        If True, only downloads the data without running the data conversion.
    rerun_preprocessing : bool, default=False
        If True, re-runs VCF to Zarr conversion even if output already exists.
    worker_processes : int, default=0
        Number of worker processes for `vcf2zarr` explode and encode steps.
        0 means use a single process (no parallelism).
    max_memory : str or None, optional
        Approximate upper bound on memory usage during the encode step (e.g. "8G", "32G").
        If None and worker_processes > 0, vcf2zarr will use its default memory limit.
        If worker_processes is 0, this argument is ignored entirely as single-process
        encoding does not require a memory bound.
    variants_chunk_size : int, default=32000
        Chunk size in the variants dimension for the output Zarr store.
    samples_chunk_size : int, default=2504
        Chunk size in the samples dimension. The 1000 Genomes Project has 2,504 samples,
        so the default stores all samples in a single chunk.

    Returns
    -------
    anndata.AnnData
        Genotype data across all autosomes in a single Zarr-backed AnnData object.

    Raises
    ------
    FileNotFoundError
        If any required VCF files are missing after download.
    RuntimeError
        If `vcf2zarr` conversion fails, e.g. due to insufficient memory or
        chunk sizes that exceed array dimensions.
    """

    return _get_1000genomes_base(
        dataset_name="1000genomes",
        config_path=config_path,
        data_home=data_home,
        verify_checksum=verify_checksum,
        only_download=only_download,
        rerun_preprocessing=rerun_preprocessing,
        worker_processes=worker_processes,
        max_memory=max_memory,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        vcf_pattern=(
            "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b."
            "20130502.genotypes.vcf.gz"
        ),
        icf_name=(
            "ALL.phase3_shapeit2_mvncall_integrated_v5b."
            "20130502.genotypes.icf"
        ),
        vcz_name=(
            "ALL.phase3_shapeit2_mvncall_integrated_v5b."
            "20130502.genotypes.vcz"
        ),
    )

def get_1000genomes_grch38(
    config_path: str = "./cellink/resources/config/1000genomes_grch38.yaml",
    data_home: str | None = None,
    verify_checksum: bool = False,
    only_download: bool = False,
    rerun_preprocessing: bool = False,
    worker_processes: int = 0,
    max_memory: str | None = None,
    variants_chunk_size: int = 32000,
    samples_chunk_size: int = 3202,
    chromosomes: list[int] | None = None,
) -> ad.AnnData | None:
    """
    Download and preprocess the GRcH38 1000 Genomes Project genotype data (https://www.cell.com/cell/pdf/S0092-8674(22)00991-6.pdf).

    This function downloads genotype files specified in a YAML configuration,
    optionally verifies checksums, converts all VCF files across chromosomes to a
    single Zarr store using `vcf2zarr`, and loads the result into an `AnnData` object
    using `cellink`.

    Parameters
    ----------
    config_path : str, default="./cellink/resources/config/1000genomes.yaml"
        Path to the YAML configuration file listing remote genotype files.
    data_home : str or None, optional
        Directory where data should be stored. If None, uses the default `cellink` data directory.
    verify_checksum : bool, default=True
        If True, verifies the checksum of downloaded files.
    only_download : bool, default=False
        If True, only downloads the data without running the data conversion.
    rerun_preprocessing : bool, default=False
        If True, re-runs VCF to Zarr conversion even if output already exists.
    worker_processes : int, default=0
        Number of worker processes for `vcf2zarr` explode and encode steps.
        0 means use a single process (no parallelism).
    max_memory : str or None, optional
        Approximate upper bound on memory usage during the encode step (e.g. "8G", "32G").
        If None and worker_processes > 0, vcf2zarr will use its default memory limit.
        If worker_processes is 0, this argument is ignored entirely as single-process
        encoding does not require a memory bound.
    variants_chunk_size : int, default=32000
        Chunk size in the variants dimension for the output Zarr store.
    samples_chunk_size : int, default=2504
        Chunk size in the samples dimension. The 1000 Genomes Project has 2,504 samples,
        so the default stores all samples in a single chunk.

    Returns
    -------
    anndata.AnnData
        Genotype data across all autosomes in a single Zarr-backed AnnData object.

    Raises
    ------
    FileNotFoundError
        If any required VCF files are missing after download.
    RuntimeError
        If `vcf2zarr` conversion fails, e.g. due to insufficient memory or
        chunk sizes that exceed array dimensions.
    """

    return _get_1000genomes_base(
        dataset_name="1000genomes_grch38",
        config_path=config_path,
        data_home=data_home,
        verify_checksum=verify_checksum,
        only_download=only_download,
        rerun_preprocessing=rerun_preprocessing,
        worker_processes=worker_processes,
        max_memory=max_memory,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        vcf_pattern=(
            "20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_"
            "chr{chrom}.recalibrated_variants.vcf.gz"
        ),
        icf_name=(
            "20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_"
            "ALL.recalibrated_variants.icf"
        ),
        vcz_name=(
            "20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_"
            "ALL.recalibrated_variants.vcz"
        ),
        chromosomes=chromosomes or range(1, 23),
    )

"""
def get_1000genomes(
    config_path: str = "./cellink/resources/config/1000genomes.yaml",
    data_home: str | None = None,
    verify_checksum: bool = True,
    only_download: bool = False,
    rerun_preprocessing: bool = False,
    worker_processes: int = 0,
    max_memory: str | None = None,
    variants_chunk_size: int = 32000,
    samples_chunk_size: int = 2504,
) -> ad.AnnData:

    data_home = get_data_home(data_home)
    DATA = data_home / "1000genomes"
    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)

    for file in config["remote_files"]:
        checksum = file.get("checksum") if verify_checksum else None
        _download_file(file["url"], DATA / file["filename"], checksum)

    if not only_download:
        vcz_path = DATA / "ALL.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcz"
        icf_path = DATA / "ALL.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.icf"

        if not os.path.isdir(vcz_path) or rerun_preprocessing:
            vcf_names = [
                f"ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
                for chrom in range(1, 23)
            ]

            for vcf_name in vcf_names:
                if not (DATA / vcf_name).exists():
                    raise FileNotFoundError(f"VCF file not found: {DATA / vcf_name}")

            vcf_files_str = " ".join(vcf_names)

            try:
                _run(
                    f"vcf2zarr explode -p {worker_processes} {vcf_files_str} {icf_path.name}",
                    cwd=DATA,
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"vcf2zarr explode failed for 1000 Genomes VCFs. "
                    f"Check that all VCF files are valid and not corrupted. Original error: {e}"
                ) from e

            encode_cmd = (
                f"vcf2zarr encode"
                f" -p {worker_processes}"
                f" -l {variants_chunk_size}"
                f" -w {samples_chunk_size}"
            )
            if worker_processes > 0 and max_memory is not None:
                encode_cmd += f" -M {max_memory}"
            encode_cmd += f" {icf_path.name} {vcz_path.name}"

            try:
                _run(encode_cmd, cwd=DATA)
            except RuntimeError as e:
                raise RuntimeError(
                    f"vcf2zarr encode failed for 1000 Genomes. "
                    f"This may be caused by insufficient memory (current limit: {max_memory}) "
                    f"or chunk sizes that exceed the array dimensions "
                    f"(variants_chunk_size={variants_chunk_size}, samples_chunk_size={samples_chunk_size}). "
                    f"Try reducing chunk sizes or increasing max_memory. Original error: {e}"
                ) from e

        gdata = cl.io.read_sgkit_zarr(vcz_path)

        return gdata

def get_1000genomes_grch38(
    config_path: str = "./cellink/resources/config/1000genomes_grch38.yaml",
    data_home: str | None = None,
    only_download: bool = False,
    verify_checksum: bool = False,
    rerun_preprocessing: bool = False,
    worker_processes: int = 0,
    max_memory: str | None = None,
    variants_chunk_size: int = 32000,
    samples_chunk_size: int = 3202,
    chromosomes: list[int] | None = None,
) -> ad.AnnData | None:

    data_home = get_data_home(data_home)
    DATA = data_home / "1000genomes_grch38"
    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)

    for file in config["remote_files"]:
        checksum = file.get("checksum") if verify_checksum else None
        _download_file(file["url"], DATA / file["filename"], checksum)

    if not only_download:
        vcz_path = DATA / "20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_ALL.recalibrated_variants.vcz" 
        icf_path = DATA / "20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_ALL.recalibrated_variants.icf" 

        if not os.path.isdir(vcz_path) or rerun_preprocessing:
            vcf_names = [
                f"20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr{chrom}.recalibrated_variants.vcf.gz" 
                for chrom in range(1, 23)
            ]

            for vcf_name in vcf_names:
                if not (DATA / vcf_name).exists():
                    raise FileNotFoundError(f"VCF file not found: {DATA / vcf_name}")

            vcf_files_str = " ".join(vcf_names)

            try:
                _run(
                    f"vcf2zarr explode -p {worker_processes} {vcf_files_str} {icf_path.name}",
                    cwd=DATA,
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"vcf2zarr explode failed for GRCh38 1000 Genomes VCFs. "
                    f"Check that all VCF files are valid and not corrupted. Original error: {e}"
                ) from e

            encode_cmd = (
                f"vcf2zarr encode"
                f" -p {worker_processes}"
                f" -l {variants_chunk_size}"
                f" -w {samples_chunk_size}"
            )
            if worker_processes > 0 and max_memory is not None:
                encode_cmd += f" -M {max_memory}"
            encode_cmd += f" {icf_path.name} {vcz_path.name}"

            try:
                _run(encode_cmd, cwd=DATA)
            except RuntimeError as e:
                raise RuntimeError(
                    f"vcf2zarr encode failed for GRCh38 1000 Genomes. "
                    f"This may be caused by insufficient memory (current limit: {max_memory}) "
                    f"or chunk sizes that exceed the array dimensions "
                    f"(variants_chunk_size={variants_chunk_size}, samples_chunk_size={samples_chunk_size}). "
                    f"Try reducing chunk sizes or increasing max_memory. Original error: {e}"
                ) from e

        gdata = cl.io.read_sgkit_zarr(vcz_path)

        return gdata
"""

def get_onek1k(
    config_path: str = "./cellink/resources/config/onek1k.yaml",
    data_home: str | None = None,
    verify_checksum: bool = True,
    only_download: bool = False,
    rerun_preprocessing: bool = False,
    worker_processes: int = 0,
    max_memory: str | None = None,
    variants_chunk_size: int = 32000,
    samples_chunk_size: int = 981,
) -> DonorData:
    """
    Download and preprocess the OneK1K genotype and expression dataset.

    This function downloads genotype and expression files listed in a YAML configuration,
    optionally verifies checksums, converts VCF files to Zarr format, performs PLINK preprocessing
    including filtering, pruning, and kinship computation, and loads the dataset into a `DonorData` object.

    Genotype preprocessing **requires PLINK** and is only performed if preprocessed
    outputs are not already present.

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
    only_download : bool, default=False
        If True, only downloads the data without running the data conversion.
    rerun_preprocessing : bool, default=False
        If True, re-runs all preprocessing steps even if outputs already exist.
    worker_processes : int, default=0
        Number of worker processes for `vcf2zarr` explode and encode steps.
        0 means use a single process (no parallelism).
    max_memory : str or None, optional
        Approximate upper bound on memory usage during the encode step (e.g. "8G", "32G").
        If None and worker_processes > 0, vcf2zarr will use its default memory limit.
        If worker_processes is 0, this argument is ignored entirely as single-process
        encoding does not require a memory bound.
    variants_chunk_size : int, default=32000
        Chunk size in the variants dimension for the output Zarr store.
    samples_chunk_size : int, default=981
        Chunk size in the samples dimension. OneK1K has 981 donors, so the default
        stores all samples in a single chunk.

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
        If preprocessing steps (VCF conversion, PLINK operations, or liftover) fail,
        e.g. due to insufficient memory or invalid chunk sizes.
    ValueError
        If variant liftover or donor alignment cannot be performed.
    EnvironmentError
        If PLINK is required for preprocessing but is not available on PATH.
    """
    data_home = get_data_home(data_home)
    DATA = data_home / "onek1k"
    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)

    for file in config["remote_files"]:
        checksum = file.get("checksum") if verify_checksum else None
        _download_file(file["url"], DATA / file["filename"], checksum)

    if not only_download:
        if not os.path.isdir(DATA / "OneK1K.noGP.vcz") or rerun_preprocessing:
            if shutil.which("plink") is None:
                raise OSError("PLINK is required for OneK1K preprocessing but was not found on PATH.")

            try:
                _run(
                    f"vcf2zarr explode -p {worker_processes} OneK1K.noGP.vcf.gz OneK1K.noGP.icf",
                    cwd=DATA,
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"vcf2zarr explode failed for OneK1K.noGP.vcf.gz. "
                    f"Check that the VCF file is valid and not corrupted. Original error: {e}"
                ) from e

            encode_cmd = (
                f"vcf2zarr encode"
                f" -p {worker_processes}"
                f" -l {variants_chunk_size}"
                f" -w {samples_chunk_size}"
            )
            if worker_processes > 0 and max_memory is not None:
                encode_cmd += f" -M {max_memory}"
            encode_cmd += " OneK1K.noGP.icf OneK1K.noGP.vcz"

            try:
                _run(encode_cmd, cwd=DATA)
            except RuntimeError as e:
                raise RuntimeError(
                    f"vcf2zarr encode failed for OneK1K.noGP. "
                    f"This may be caused by insufficient memory (current limit: {max_memory}) "
                    f"or chunk sizes that exceed the array dimensions "
                    f"(variants_chunk_size={variants_chunk_size}, samples_chunk_size={samples_chunk_size}). "
                    f"Try reducing chunk sizes or increasing max_memory. Original error: {e}"
                ) from e

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
        gdata.obsm["gPCs"].columns = gdata.obsm["gPCs"].columns.astype(str)

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

def get_dummy_onek1k(
    config_path: str = "./cellink/resources/config/dummy_onek1k.yaml",
    data_home: str | None = None,
    verify_checksum: bool = True,
) -> DonorData:
    """
    Download and load the dummy OneK1K dataset.

    This function downloads a pre-processed subset of the OneK1K dataset containing:
    - Full chromosome 22 genotype data
    - 0.1% sample of SNPs from chromosomes 1-21
    - ~100 donors (randomly sampled)
    - All cell types and expression data preserved
    - **Gene annotations pre-included** (GAnn.start, GAnn.end, GAnn.chrom)
    - **Only QTL-relevant genes** (chr 1-22, within ±1Mb of any SNP)

    The dummy dataset is ideal for tutorials, testing, and demonstrations as it is
    ~100x smaller than the full OneK1K dataset while maintaining the same structure
    and API. Unlike the full dataset, gene annotations are already included,
    eliminating the need for pybiomart calls in tutorials.

    Parameters
    ----------
    config_path : str, default="./cellink/resources/config/dummy_onek1k.yaml"
        Path to the YAML configuration file listing the remote dummy dataset file.
    data_home : str or None, optional
        Directory where data should be stored. If None, uses the default `cellink`
        data directory.
    verify_checksum : bool, default=True
        If True, verifies the checksum of the downloaded file.

    Returns
    -------
    cellink.DonorData
        A `DonorData` object containing preprocessed genotype (`G`) and expression
        (`C`) data, along with kinship and principal component metadata. Gene
        annotations are already included in `dd.C.var`.

    Examples
    --------
    >>> from cellink.resources import get_dummy_onek1k
    >>> dd = get_dummy_onek1k()
    >>> print(dd.shape)  # (n_donors, n_snps, n_cells, n_genes)
    >>> # Gene annotations are already included!
    >>> print(dd.C.var[[GAnn.start, GAnn.end, GAnn.chrom]].head())

    Notes
    -----
    The dummy dataset maintains the same structure as the full OneK1K dataset:
    - dd.G: Genotype data (donors x SNPs)
    - dd.C: Single-cell expression data (cells x genes)
    - dd.G.obsm["gPCs"]: Genotype principal components
    - dd.G.uns["kinship"]: Kinship matrix
    - dd.C.var[GAnn.start/end/chrom]: Gene annotations (pre-included!)

    The dataset is provided as a single file (.dd.h5 or .dd.zarr) that can be
    quickly downloaded and loaded without additional preprocessing.

    Key differences from full dataset:
    - Gene annotations are pre-included (no pybiomart needed)
    - Only genes on chromosomes 1-22 are included
    - Only genes within ±1Mb of any SNP (QTL-relevant)
    - Reduces gene count from ~20k to ~5-10k for faster processing

    See Also
    --------
    get_onek1k : Load the full OneK1K dataset
    """
    data_home = get_data_home(data_home)
    DATA = data_home / "dummy_onek1k"
    DATA.mkdir(exist_ok=True)

    config = _load_config(config_path)

    file_info = config["remote_files"][0]
    filename = file_info["filename"]
    checksum = file_info.get("checksum") if verify_checksum else None

    file_path = DATA / filename
    _download_file(file_info["url"], file_path, checksum)

    if filename.endswith(".dd.h5"):
        dd = read_h5_dd(str(file_path))
    elif filename.endswith(".dd.zarr"):
        dd = read_zarr_dd(str(file_path))
    else:
        raise ValueError(f"Unsupported file format: {filename}")

    logging.info(f"Loaded dummy OneK1K dataset: {dd.shape}")

    return dd


if __name__ == "__main__":
    get_dummy_onek1k()
    get_onek1k()
    get_1000genomes()
    get_1000genomes_grch38()
