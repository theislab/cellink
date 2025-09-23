from typing import Literal

from anndata import AnnData
import anndata as ad
from muon import MuData
import numpy as np
import pandas as pd

from cellink._core.data_fields import CAnn, DAnn, GAnn, VAnn

N_DONORS = 10
N_GENES = 20
N_PEAKS = 50
N_SNPS = 5


CELLTYPES = ["CD4", "CD8", "NK"]
MIN_N_CELLS = 10
MAX_N_CELLS = 100
MIN_GENE_LENGTH = 18_000
MAX_GENE_LENGTH = 30_000
EXAMPLE_CHROMOSOME = 1
DUMMY_COVARIATES = ["cov1", "cov2", "cov3"]
CELL_PREFIX = "C"
DONOR_PREFIX = "D"
GENE_PREFIX = "G"
SNP_PREFIX = "SNP"


def _sim_donor(
    start_index: int = None,
    n_cells: int = None,
    n_genes: int = None,
    has_all_celltypes: bool = False,
    strategy: Literal["randn", "poisson", "negative_binomial", "binomial", "uniform"] = "randn",
    mean_nb: int = 5,
    dispersion_nb: int = 2,
    p_binomial: float = 0.1,
) -> AnnData:
    """
    Simulate a single donor's cell x gene expression matrix as an AnnData object.

    Generates synthetic expression data for one donor using different statistical distributions,
    attaches random covariates, and annotates genes with basic genomic features.

    Parameters
    ----------
    start_index : int
        Starting cell index (used to build unique cell IDs).
    n_cells : int
        Number of cells to simulate for the donor.
    n_genes : int
        Number of genes to simulate.
    has_all_celltypes : bool
        Whether to include all cell types in `CELLTYPES`. If False, the last cell type is excluded.
    strategy : {'randn', 'poisson', 'negative_binomial', 'binomial', 'uniform'}, default='randn'
        Distribution used to simulate expression counts:
        - 'randn': standard normal
        - 'poisson': Poisson with λ=5
        - 'negative_binomial': Negative binomial with parameters (`mean_nb`, `dispersion_nb`)
        - 'binomial': Binomial with n=1 and probability `p_binomial`
        - 'uniform': Uniform(0,1)
    mean_nb : int, default=5
        Mean count for negative binomial distribution.
    dispersion_nb : int, default=2
        Dispersion parameter for negative binomial distribution.
    p_binomial : float, default=0.1
        Success probability for binomial distribution.

    Returns
    -------
    AnnData
        AnnData object with:
        - `X` : simulated expression matrix (n_cells x n_genes)
        - `obs` : DataFrame with columns:
            - celltype
            - dummy covariates (`cov1`, `cov2`, `cov3`)
        - `var` : DataFrame with columns:
            - gene name, chromosome, start, end, strand
    """
        
    if strategy == "randn":
        X = np.random.randn(n_cells, n_genes)
    elif strategy == "poisson":
        X = np.random.poisson(lam=5, size=(n_cells, n_genes))
    elif strategy == "negative_binomial":
        r = 1 / dispersion_nb
        p = mean_nb / (mean_nb + r)
        X = np.random.negative_binomial(n=r, p=p, size=(n_cells, n_genes))
    elif strategy == "binomial":
        X = np.random.binomial(n=1, p=p_binomial, size=(n_cells, n_genes))
    elif strategy == "uniform":
        X = np.random.uniform(size=(n_cells, n_genes))
    _celltypes = CELLTYPES if has_all_celltypes else CELLTYPES[:-1]
    obs = pd.DataFrame({CAnn.celltype: np.random.choice(_celltypes, size=n_cells)})
    obs.index = [f"{CELL_PREFIX}{i}" for i in range(start_index, start_index + n_cells)]
    obs[DUMMY_COVARIATES] = np.random.randn(n_cells, len(DUMMY_COVARIATES))
    gene_lengths = np.random.randint(MIN_GENE_LENGTH, MAX_GENE_LENGTH, size=n_genes)
    var = pd.DataFrame(
        {
            GAnn.name: np.array([f"{GENE_PREFIX}{i}" for i in np.arange(n_genes)]),
            GAnn.chrom: EXAMPLE_CHROMOSOME,
            GAnn.start: np.concatenate([[0], np.cumsum(gene_lengths[:-1])]),
            GAnn.end: np.cumsum(gene_lengths),
            GAnn.strand: np.random.choice([1, -1], size=n_genes),
        }
    ).set_index(GAnn.name)
    return AnnData(X=X, obs=obs, var=var)


def sim_mudata(
    n_donors: int = N_DONORS, n_genes: int = N_GENES, n_peaks: int = N_PEAKS, min_n_cells: int = MIN_N_CELLS, max_n_cells: int = MAX_N_CELLS
) -> MuData:
    """
    Simulate a multi-modal `MuData` object with RNA and ATAC layers.

    This function generates paired RNA-seq and ATAC-seq data across multiple donors,
    combining them into a single `MuData` container. Cell-level metadata is harmonized
    so that both modalities share the same donor and cell type assignments.

    Parameters
    ----------
    n_donors : int, default=N_DONORS
        Number of donors to simulate.
    n_genes : int, default=N_GENES
        Number of genes in the RNA modality.
    n_peaks : int, default=N_PEAKS
        Number of peaks in the ATAC modality.
    min_n_cells : int, default=MIN_N_CELLS
        Minimum number of cells per donor.
    max_n_cells : int, default=MAX_N_CELLS
        Maximum number of cells per donor.

    Returns
    -------
    MuData
        MuData object with modalities:
        - 'rna': AnnData of simulated RNA counts (negative binomial)
        - 'atac': AnnData of simulated ATAC accessibility (binomial)
        Shared `obs` fields include:
        - 'celltype'
        - 'donor_id'
    """
        
    rna = sim_adata(
        n_donors=N_DONORS,
        n_genes=N_GENES,
        min_n_cells=MIN_N_CELLS,
        max_n_cells=MAX_N_CELLS,
        strategy="negative_binomial",
    )
    atac = sim_adata(
        n_donors=N_DONORS, n_genes=N_PEAKS, min_n_cells=MIN_N_CELLS, max_n_cells=MAX_N_CELLS, strategy="binomial"
    )
    adata = MuData({"rna": rna, "atac": atac})
    adata.obs["celltype"] = adata.obs["rna:celltype"]
    adata.obs["donor_id"] = adata.obs["rna:donor_id"]
    return adata


def sim_adata(
    n_donors: int = N_DONORS,
    n_genes: int = N_GENES,
    min_n_cells: int = MIN_N_CELLS,
    max_n_cells: int = MAX_N_CELLS,
    strategy: Literal["randn", "poisson", "negative_binomial", "binomial", "uniform"] = "randn",
) -> AnnData:
    """
    Simulate an AnnData object across multiple donors.

    Each donor contributes a random number of cells (between `min_n_cells` and `max_n_cells`)
    with expression values simulated according to the chosen distribution.
    The first donor is intentionally missing one cell type, while all others include
    the full set of cell types.

    Parameters
    ----------
    n_donors : int, default=N_DONORS
        Number of donors to simulate.
    n_genes : int, default=N_GENES
        Number of genes to simulate.
    min_n_cells : int, default=MIN_N_CELLS
        Minimum number of cells per donor.
    max_n_cells : int, default=MAX_N_CELLS
        Maximum number of cells per donor.
    strategy : {'randn', 'poisson', 'negative_binomial', 'binomial', 'uniform'}, default='randn'
        Distribution to use for simulating expression counts.

    Returns
    -------
    AnnData
        Concatenated AnnData object containing all donors with:
        - `obs` : cell-level metadata including donor ID and celltype
        - `var` : gene-level metadata including chromosome, start, end
    """

    adatas = []
    has_all_celltypes = [True] * (n_donors - 1) + [False]
    cum_n_cells = 0
    for i in range(n_donors):
        n_cells = np.random.randint(min_n_cells, max_n_cells)
        has_all_celltypes = i != 0  # the first donor misses one celltype
        adatas.append(_sim_donor(cum_n_cells, n_cells, n_genes, has_all_celltypes, strategy=strategy))
        cum_n_cells += n_cells
    donors = [f"{DONOR_PREFIX}{i}" for i in range(n_donors)]
    adata = ad.concat(adatas, merge="first", keys=donors, label=DAnn.donor)
    return adata


def sim_gdata(
    n_donors: int = N_DONORS, 
    n_snps: int = N_SNPS, 
    adata: AnnData | None = None
) -> AnnData:
    """
    Simulate genotype data as an AnnData object.

    Generates SNP genotypes for multiple donors using binomial sampling
    according to randomly assigned minor allele frequencies (MAFs).
    Optionally aligns SNP positions to overlap with gene coordinates from an RNA AnnData.

    Parameters
    ----------
    n_donors : int, default=N_DONORS
        Number of donors to simulate.
    n_snps : int, default=N_SNPS
        Number of SNPs to simulate if `adata` is None.
    adata : AnnData, optional
        If provided, SNP positions are aligned with the genes in `adata.var`
        such that SNPs fall within gene intervals.

    Returns
    -------
    AnnData
        AnnData object with:
        - `X` : genotype matrix (n_donors x n_snps), values in {0,1,2}
        - `obs` : donor IDs
        - `var` : SNP metadata including chromosome, position, alleles, and MAF
    """

    if adata is None:
        pos = np.arange(1, n_snps + 1)
    else:
        # make sure the variants overlap with the genes
        pos = []
        for _, row in adata.var.iterrows():
            _pos = np.arange(N_SNPS) - (N_SNPS // 2) + row[GAnn.start]
            _pos = _pos[(_pos >= 0) & (_pos < row[GAnn.end])]
            pos.append(_pos)
        pos = np.concatenate(pos)
        n_snps = len(pos)
    mafs = np.random.rand(n_snps)
    X = np.random.binomial(n=2, p=mafs, size=(n_donors, n_snps))
    obs = pd.DataFrame(index="D" + pd.RangeIndex(n_donors, name=DAnn.donor).astype(str))

    var = pd.DataFrame(
        {
            VAnn.chrom: EXAMPLE_CHROMOSOME,
            VAnn.pos: pos,
            VAnn.a0: np.random.choice(["A", "T"], size=n_snps),
            VAnn.a1: np.random.choice(["C", "G"], size=n_snps),
            VAnn.maf: mafs,
        },
        index=SNP_PREFIX + pd.RangeIndex(n_snps).astype(str),
    )
    var.index.name = VAnn.index
    gdata = AnnData(X=X, obs=obs, var=var)
    return gdata
