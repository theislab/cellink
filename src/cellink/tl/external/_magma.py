import logging
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Literal

import pandas as pd
import yaml

from cellink._core import DonorData
from cellink.io import to_plink
from cellink.resources._utils import _download_file, get_data_home

logger = logging.getLogger(__name__)

def load_magma_config(config_file: str) -> dict:
    """
    Load MAGMA reference file URLs from YAML configuration.

    Parameters
    ----------
    config_file : str
        Path to YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary with reference file information.
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config

def get_gene_id_mapping(
    genome_build: str = "GRCh38",
    gene_id_type: Literal["entrez", "ensembl", "gene_name"] = "ensembl"
) -> pd.DataFrame:
    """
    Get gene ID mapping from Ensembl using pybiomart.

    Parameters
    ----------
    genome_build : str
        Genome build version ('GRCh37' or 'GRCh38').
    gene_id_type : str
        Type of gene ID to use as index: 'entrez', 'ensembl', or 'gene_name'.

    Returns
    -------
    pd.DataFrame
        DataFrame with gene ID mappings indexed by the chosen ID type.
    """
    try:
        from pybiomart import Server
    except ImportError:
        raise ImportError(
            "pybiomart is required for gene ID mapping. "
            "Install it with: pip install pybiomart"
        )

    logger.info(f"Fetching gene ID mappings from Ensembl ({genome_build})")

    if genome_build == "GRCh37":
        host = 'http://grch37.ensembl.org'
    elif genome_build == "GRCh38":
        host = 'http://www.ensembl.org'
    else:
        raise ValueError(f"Invalid genome_build: {genome_build}")

    server = Server(host=host)
    dataset = server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl']
    mapping_df = dataset.query(
        attributes=['ensembl_gene_id', 'external_gene_name', 'entrezgene_id']
    )

    mapping_df = mapping_df.rename(columns={"Gene stable ID": "ensembl_gene_id",
                                            "Gene name": "external_gene_name",
                                            "NCBI gene (formerly Entrezgene) ID": "entrezgene_id"})

    mask = ~mapping_df["entrezgene_id"].isna()
    mapping_df_filtered = mapping_df[mask]
    mapping_df = mapping_df_filtered[
        ~mapping_df_filtered["ensembl_gene_id"].duplicated(keep=False)
    ]

    mapping_df = mapping_df.dropna(subset=['entrezgene_id'])
    mapping_df['entrezgene_id'] = mapping_df['entrezgene_id'].astype(int)

    if gene_id_type == "entrez":
        mapping_df = mapping_df.set_index('entrezgene_id')
    elif gene_id_type == "ensembl":
        mapping_df = mapping_df.set_index('ensembl_gene_id')
    elif gene_id_type == "gene_name":
        mapping_df = mapping_df.set_index('external_gene_name')
    else:
        raise ValueError(
            f"Invalid gene_id_type: {gene_id_type}. "
            "Options: 'entrez', 'ensembl', 'gene_name'"
        )

    logger.info(f"Retrieved {len(mapping_df)} gene mappings")
    return mapping_df


def convert_gene_loc_file(
    gene_loc_file: Path,
    output_file: Path,
    genome_build: str = "GRCh38",
    gene_id_type: Literal["entrez", "ensembl", "gene_name"] = "ensembl"
) -> Path:
    """
    Convert gene location file to use different gene ID types.

    The original MAGMA gene location files use Entrez Gene IDs. This function
    converts them to use Ensembl Gene IDs or Gene Names if desired.

    Parameters
    ----------
    gene_loc_file : Path
        Path to original gene location file (with Entrez IDs).
    output_file : Path
        Path for output converted file.
    genome_build : str
        Genome build version ('GRCh37' or 'GRCh38').
    gene_id_type : str
        Target gene ID type: 'entrez', 'ensembl', or 'gene_name'.

    Returns
    -------
    Path
        Path to the converted gene location file.
    """
    if gene_id_type == "entrez":
        logger.info("Using original Entrez Gene IDs")
        return gene_loc_file

    logger.info(f"Converting gene location file to {gene_id_type} IDs")

    gene_loc = pd.read_csv(
        gene_loc_file,
        sep=r'\s+',
        header=None,
        names=['entrez_id', 'chr', 'start', 'end', 'strand', 'gene_name']
    )

    mapping_df = get_gene_id_mapping(genome_build=genome_build, gene_id_type="entrez")

    if gene_id_type == "ensembl":
        gene_loc = gene_loc.merge(
            mapping_df[['ensembl_gene_id']],
            left_on='entrez_id',
            right_index=True,
            how='left'
        )
        gene_loc = gene_loc.dropna(subset=['ensembl_gene_id'])
        gene_loc['new_id'] = gene_loc['ensembl_gene_id']
    elif gene_id_type == "gene_name":
        gene_loc['new_id'] = gene_loc['gene_name']

    output_df = gene_loc[['new_id', 'chr', 'start', 'end', 'strand', 'gene_name']]
    output_df.columns = ['gene_id', 'chr', 'start', 'end', 'strand', 'gene_name']

    output_df.to_csv(output_file, sep='\t', header=False, index=False)

    logger.info(
        f"Converted {len(output_df)} genes from Entrez to {gene_id_type} IDs"
    )
    logger.info(f"Converted gene location file: {output_file}")

    return output_file

def download_magma_references(
    genome_build: str = "GRCh38",
    reference_panel: str = None,
    gene_id_type: Literal["entrez", "ensembl", "gene_name"] = "ensembl",
    config_file: str = "configs/magma.yaml",
    data_home: str | Path | None = None,
) -> Tuple[Path, Path]:
    """
    Download MAGMA reference files (gene locations and optionally LD reference panel).

    This function downloads the gene location file and optionally a 1000 Genomes
    reference panel for LD estimation. Files are cached locally and only downloaded
    once.

    Parameters
    ----------
    genome_build : str, default='GRCh38'
        Genome build version. Options: 'GRCh37' (hg19) or 'GRCh38' (hg38).
    reference_panel : str, optional
        Population reference panel for LD estimation. Options: 'EUR', 'EAS', 'AFR'.
        If None, no reference panel is downloaded (useful when using your own genotypes).
    gene_id_type : str, default='ensembl'
        Gene ID type to use: 'entrez', 'ensembl', or 'gene_name'.
        Entrez IDs are the original format; conversion to other types requires pybiomart.
    config_file : str, default='configs/magma.yaml'
        Path to YAML configuration file with reference URLs.
    data_home : str or Path, optional
        Directory to store downloaded files. If None, uses cellink's default data directory.

    Returns
    -------
    tuple of Path
        (gene_loc_file, reference_prefix)
        - gene_loc_file: Path to the gene location file
        - reference_prefix: Path prefix for reference panel files (or None if not downloaded)

    Examples
    --------
    >>> # Download gene locations with Ensembl IDs
    >>> gene_loc, _ = download_magma_references(
    ...     genome_build="GRCh38",
    ...     gene_id_type="ensembl"
    ... )

    >>> # Download gene locations and EUR reference panel
    >>> gene_loc, ref_prefix = download_magma_references(
    ...     genome_build="GRCh38",
    ...     reference_panel="EUR",
    ...     gene_id_type="gene_name"
    ... )

    Notes
    -----
    For lightweight analyses or tutorials, it's recommended to:
    1. Download only the gene location file (reference_panel=None)
    2. Use genotypes from your DonorData object as the LD reference

    This avoids downloading large reference panels (~1GB) and uses your actual
    study population for LD estimation, which may be more appropriate.
    """
    data_home = get_data_home(data_home)
    magma_dir = data_home / "magma_references"
    magma_dir.mkdir(exist_ok=True, parents=True)

    config = load_magma_config(config_file)

    if genome_build not in config["gene_loc"]:
        raise ValueError(
            f"Invalid genome_build: {genome_build}. "
            f"Options: {list(config['gene_loc'].keys())}"
        )

    gene_loc_info = config["gene_loc"][genome_build]
    gene_loc_zip = magma_dir / f"{genome_build}.zip"
    gene_loc_file = magma_dir / gene_loc_info["filename"]

    if not gene_loc_file.exists():
        logger.info(f"Downloading {gene_loc_info['description']}")
        _download_file(gene_loc_info["url"], gene_loc_zip)

        import zipfile

        logger.info(f"Extracting {gene_loc_zip}")
        with zipfile.ZipFile(gene_loc_zip, "r") as zip_ref:
            zip_ref.extractall(magma_dir)

        gene_loc_zip.unlink()
        logger.info(f"Gene location file ready: {gene_loc_file}")
    else:
        logger.info(f"Using cached gene location file: {gene_loc_file}")

    if gene_id_type != "entrez":
        converted_file = magma_dir / f"{genome_build}_{gene_id_type}.gene.loc"
        if not converted_file.exists():
            gene_loc_file = convert_gene_loc_file(
                gene_loc_file=gene_loc_file,
                output_file=converted_file,
                genome_build=genome_build,
                gene_id_type=gene_id_type
            )
        else:
            logger.info(f"Using cached converted gene location file: {converted_file}")
            gene_loc_file = converted_file

    ref_prefix = None
    if reference_panel is not None:
        if reference_panel not in config["reference_data"]:
            raise ValueError(
                f"Invalid reference_panel: {reference_panel}. "
                f"Options: {list(config['reference_data'].keys())}"
            )

        ref_info = config["reference_data"][reference_panel]
        ref_zip = magma_dir / f"g1000_{reference_panel.lower()}.zip"
        ref_prefix = magma_dir / f"g1000_{reference_panel.lower()}"

        all_exist = all((magma_dir / f).exists() for f in ref_info["files"])

        if not all_exist:
            logger.info(f"Downloading {ref_info['description']}")
            logger.warning(
                f"Reference panel is large (~1GB). Consider using your own genotypes instead."
            )
            _download_file(ref_info["url"], ref_zip)

            import zipfile

            logger.info(f"Extracting {ref_zip}")
            with zipfile.ZipFile(ref_zip, "r") as zip_ref:
                zip_ref.extractall(magma_dir)

            ref_zip.unlink()
            logger.info(f"Reference panel ready: {ref_prefix}")
        else:
            logger.info(f"Using cached reference panel: {ref_prefix}")

    return gene_loc_file, ref_prefix

def prepare_magma_inputs_from_dd(
    dd: DonorData,
    gwas_sumstats: pd.DataFrame,
    output_prefix: str = "magma_input",
    genome_build: str = "GRCh38",
    gene_id_type: Literal["entrez", "ensembl", "gene_name"] = "ensembl",
    use_genotypes_for_ld: bool = True,
    col_mapping: dict = None,
    config_file: str = "configs/magma.yaml",
) -> Tuple[Path, Path, Path, Path]:
    """
    Prepare all input files needed for MAGMA analysis using DonorData genotypes.

    This function creates:
    1. SNP location file from GWAS summary statistics
    2. P-value file from GWAS summary statistics
    3. PLINK files from DonorData genotypes (for LD reference)
    4. Downloads gene location file if needed

    Parameters
    ----------
    dd : DonorData
        DonorData object containing genotype data (dd.G).
    gwas_sumstats : pd.DataFrame
        GWAS summary statistics with columns: ['SNP', 'CHR', 'BP', 'P']
        or ['rsID', 'Chromosome', 'BP', 'P.value'].
    output_prefix : str, default='magma_input'
        Prefix for output files.
    genome_build : str, default='GRCh38'
        Genome build version ('GRCh37' or 'GRCh38').
    gene_id_type : str, default='ensembl'
        Gene ID type to use: 'entrez', 'ensembl', or 'gene_name'.
    use_genotypes_for_ld : bool, default=True
        If True, export DonorData genotypes as PLINK files for LD reference.
        If False, you'll need to provide a reference panel separately.
    col_mapping : dict, optional
        Dictionary to map column names in GWAS summary statistics to standard names.
        Default maps common variants of SNP, CHR, BP, and P column names.
        Example: {"variant_id": "SNP", "chromosome": "CHR", "p_value": "P"}
    config_file : str, default='configs/magma.yaml'
        Path to YAML configuration file with reference URLs.

    Returns
    -------
    tuple of Path
        (snp_loc_file, pval_file, gene_loc_file, ld_reference_prefix)

    Examples
    --------
    >>> # Prepare all inputs using your genotypes with Ensembl gene IDs
    >>> snp_loc, pval, gene_loc, ld_ref = prepare_magma_inputs_from_dd(
    ...     dd,
    ...     gwas_df,
    ...     output_prefix="trait_magma",
    ...     gene_id_type="ensembl"
    ... )
    >>>

    Notes
    -----
    This function is designed to make MAGMA analysis easy with cellink's DonorData:
    - Uses your actual genotypes for LD estimation 
    - Handles all file conversions automatically
    - Downloads only the necessary gene location file
    - Supports different gene ID types (Entrez, Ensembl, Gene Name)
    """
    logger.info("Preparing MAGMA input files from DonorData")

    output_dir = Path(output_prefix).parent if Path(output_prefix).parent != Path('.') else Path.cwd()
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Downloading/checking gene location file")
    gene_loc_file, _ = download_magma_references(
        genome_build=genome_build,
        gene_id_type=gene_id_type,
        reference_panel=None,  
        config_file=config_file,
    )

    logger.info("Preparing SNP location and p-value files")

    if col_mapping is None:
        col_mapping = {
            "rsID": "SNP",
            "variant_id": "SNP",
            "chr": "CHR",
            "pval": "P",
        }

    identifier_columns = ["variant_id", "rsID", "SNP"]
    has_valid_identifier = any(
        col in gwas_sumstats.columns and gwas_sumstats[col].notna().any()
        for col in identifier_columns
    )
    if not has_valid_identifier:
        gwas_sumstats["SNP"] = (
            gwas_sumstats["chromosome"].astype(str)
            + "_"
            + gwas_sumstats["base_pair_location"].astype(str)
            + "_"
            + gwas_sumstats["effect_allele"].astype(str)
            + "_"
            + gwas_sumstats["other_allele"].astype(str)
        )

    df = gwas_sumstats.copy()
    df = df.rename(columns=col_mapping)

    required_cols = ["SNP", "CHR", "BP", "P"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in GWAS summary statistics: {missing_cols}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=required_cols)

    snp_loc_file = Path(f"{output_prefix}.snp_loc.txt")
    df[["SNP", "CHR", "BP"]].to_csv(snp_loc_file, sep="\t", header=False, index=False)
    logger.info(f"Created SNP location file: {snp_loc_file}")

    pval_file = Path(f"{output_prefix}.p_val.txt")
    df[["SNP", "P"]].to_csv(pval_file, sep="\t", header=False, index=False)
    logger.info(f"Created p-value file: {pval_file}")

    ld_ref_prefix = None
    if use_genotypes_for_ld:
        logger.info("Exporting genotypes to PLINK format for LD reference")
        ld_ref_prefix = Path(f"{output_prefix}_ld_ref")

        import numpy as np

        maf = dd.G.X.sum(axis=0) / (2 * dd.G.n_obs)
        maf = np.minimum(maf, 1 - maf)
        common_variants = maf > 0.01

        if common_variants.sum() < len(maf):
            logger.info(
                f"Filtering to {common_variants.sum()} common variants (MAF > 0.01) "
                f"from {len(maf)} total variants"
            )
            gdata_filtered = dd.G[:, common_variants].copy()
        else:
            gdata_filtered = dd.G

        to_plink(gdata_filtered, str(ld_ref_prefix))
        logger.info(f"Created PLINK files: {ld_ref_prefix}.{{bed,bim,fam}}")

    return snp_loc_file, pval_file, gene_loc_file, ld_ref_prefix


def run_magma_annotation(
    snp_loc_file: Path,
    gene_loc_file: Path,
    output_prefix: str,
    window_size: Tuple[int, int] = (35, 10),
    magma_bin: str = "magma",
) -> Path:
    """
    Run MAGMA SNP annotation to genes.

    Parameters
    ----------
    snp_loc_file : Path
        Path to SNP location file (3 columns: SNP, CHR, BP).
    gene_loc_file : Path
        Path to gene location file.
    output_prefix : str
        Prefix for output files.
    window_size : tuple of int, default=(35, 10)
        Window around genes in kilobases (upstream, downstream).
    magma_bin : str, default='magma'
        Path to MAGMA binary. Can be 'magma' if in PATH, or a full path like './magma/magma'.

    Returns
    -------
    Path
        Path to the annotation file (.genes.annot).

    Raises
    ------
    RuntimeError
        If MAGMA command fails.
    """
    logger.info("Running MAGMA SNP annotation")

    annotation_file = Path(f"{output_prefix}.genes.annot")

    cmd = [
        magma_bin,
        "--annotate",
        f"window={window_size[0]},{window_size[1]}",
        "--snp-loc", str(snp_loc_file),
        "--gene-loc", str(gene_loc_file),
        "--out", output_prefix,
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"MAGMA annotation failed: {e.stderr}")
        raise RuntimeError(f"MAGMA annotation failed: {e.stderr}")

    logger.info(f"Annotation complete: {annotation_file}")
    return annotation_file


def run_magma_gene_analysis(
    pval_file: Path,
    annotation_file: Path,
    ld_reference_prefix: Path,
    output_prefix: str,
    n_samples: int = None,
    magma_bin: str = "magma",
) -> Path:
    """
    Run MAGMA gene analysis to compute gene-level statistics.

    Parameters
    ----------
    pval_file : Path
        Path to p-value file (2 columns: SNP, P).
    annotation_file : Path
        Path to annotation file from MAGMA annotate step.
    ld_reference_prefix : Path
        Prefix for PLINK files to use as LD reference.
    output_prefix : str
        Prefix for output files.
    n_samples : int, optional
        Sample size for the GWAS. If not provided, MAGMA will try to infer it.
    magma_bin : str, default='magma'
        Path to MAGMA binary. Can be 'magma' if in PATH, or a full path like './magma/magma'.

    Returns
    -------
    Path
        Path to the gene analysis results file (.genes.out).

    Raises
    ------
    RuntimeError
        If MAGMA command fails.
    """
    logger.info("Running MAGMA gene analysis")

    results_file = Path(f"{output_prefix}.genes.out")

    cmd = [
        magma_bin,
        "--bfile", str(ld_reference_prefix),
        "--pval", str(pval_file), "N=", str(n_samples),
        "--gene-annot", str(annotation_file),
        "--out", f"{output_prefix}",
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"MAGMA gene analysis failed: {e.stderr}")
        raise RuntimeError(f"MAGMA gene analysis failed: {e.stderr}")

    logger.info(f"Gene analysis complete: {results_file}")
    return results_file

def run_magma_pipeline(
    dd: DonorData,
    gwas_sumstats: pd.DataFrame,
    output_prefix: str = "magma_results",
    genome_build: str = "GRCh38",
    gene_id_type: Literal["entrez", "ensembl", "gene_name"] = "ensembl",
    window_size: Tuple[int, int] = (35, 10),
    n_samples: int = None,
    col_mapping: dict = None,
    config_file: str = "configs/magma.yaml",
    magma_bin: str = "magma",
) -> Path:
    """
    Complete MAGMA pipeline: prepare inputs, annotate, and analyze.

    This is a convenience function that runs the complete MAGMA workflow
    using genotypes from DonorData as the LD reference.

    Parameters
    ----------
    dd : DonorData
        DonorData object containing genotype data.
    gwas_sumstats : pd.DataFrame
        GWAS summary statistics with columns: ['SNP', 'CHR', 'BP', 'P'].
    output_prefix : str, default='magma_results'
        Prefix for all output files.
    genome_build : str, default='GRCh37'
        Genome build version ('GRCh37' or 'GRCh38').
    gene_id_type : str, default='ensembl'
        Gene ID type to use: 'entrez', 'ensembl', or 'gene_name'.
    window_size : tuple of int, default=(35, 10)
        Window around genes in kilobases (upstream, downstream).
    n_samples : int
        Sample size for the GWAS.
    col_mapping : dict, optional
        Dictionary to map column names in GWAS summary statistics.
        If None, uses default mapping for common column name variants.
    config_file : str, default='configs/magma.yaml'
        Path to YAML configuration file with reference URLs.
    magma_bin : str, default='magma'
        Path to MAGMA binary. Can be 'magma' if in PATH, or a full path like './magma/magma'.

    Returns
    -------
    Path
        Path to the final gene-level results file (.genes.out).

    Examples
    --------
    >>> from cellink.resources import get_dummy_onek1k
    >>> from cellink.tl.external import run_magma_pipeline
    >>>
    >>> # Load data
    >>> dd = get_dummy_onek1k()
    >>> gwas_df = pd.read_csv("gwas_sumstats.txt", sep="\t")
    >>>
    >>> # Run complete MAGMA pipeline with Ensembl gene IDs
    >>> results_file = run_magma_pipeline(
    ...     dd,
    ...     gwas_df,
    ...     output_prefix="t2d_magma",
    ...     gene_id_type="ensembl",
    ...     n_samples=100000,
    ...     magma_bin="./magma/magma"
    ... )
    >>>
    >>> # Load results
    >>> magma_results = pd.read_csv(results_file, sep=r'\s+')
    """
    logger.info("Starting MAGMA pipeline")

    snp_loc, pval, gene_loc, ld_ref = prepare_magma_inputs_from_dd(
        dd=dd,
        gwas_sumstats=gwas_sumstats,
        output_prefix=output_prefix,
        genome_build=genome_build,
        gene_id_type=gene_id_type,
        use_genotypes_for_ld=True,
        col_mapping=col_mapping,
        config_file=config_file,
    )

    annotation_file = run_magma_annotation(
        snp_loc_file=snp_loc,
        gene_loc_file=gene_loc,
        output_prefix=output_prefix,
        window_size=window_size,
        magma_bin=magma_bin,
    )

    results_file = run_magma_gene_analysis(
        pval_file=pval,
        annotation_file=annotation_file,
        ld_reference_prefix=ld_ref,
        output_prefix=output_prefix,
        n_samples=n_samples,
        magma_bin=magma_bin,
    )

    logger.info(f"MAGMA pipeline complete! Results: {results_file}")

    return results_file
