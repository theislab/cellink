import anndata as ad
import numpy as np
import pandas as pd

from cellink._core.annotation import CAnn, DAnn, GAnn, VAnn

N_DONORS = 10
N_GENES = 20
N_SNPS = 5


CELLTYPES = ["CD4", "CD8", "NK"]
MIN_N_CELLS = 10
MAX_N_CELLS = 100
MIN_GENE_LENGTH = 18_000
MAX_GENE_LENGTH = 30_000
EXAMPLE_CHROMOSOME = 1
DUMMY_COVARIATES = ["cov1", "cov2", "cov3"]


def _sim_donor(start_index, n_cells, n_genes, has_all_celltypes):
    X = np.random.randn(n_cells, n_genes)
    _celltypes = CELLTYPES if has_all_celltypes else CELLTYPES[:-1]
    obs = pd.DataFrame({CAnn.celltype: np.random.choice(_celltypes, size=n_cells)})
    obs.index = [f"C{i}" for i in range(start_index, start_index + n_cells)]
    obs[DUMMY_COVARIATES] = np.random.randn(n_cells, len(DUMMY_COVARIATES))
    gene_lengths = np.random.randint(MIN_GENE_LENGTH, MAX_GENE_LENGTH, size=n_genes)
    var = pd.DataFrame(
        {
            GAnn.name: np.array([f"G{i}" for i in np.arange(n_genes)]),
            GAnn.chrom: EXAMPLE_CHROMOSOME,
            GAnn.start: np.concatenate([[0], np.cumsum(gene_lengths[:-1])]),
            GAnn.end: np.cumsum(gene_lengths),
            GAnn.strand: np.random.choice([1, -1], size=n_genes),
        }
    ).set_index(GAnn.name)
    return ad.AnnData(X=X, obs=obs, var=var)


def sim_adata(
    n_donors=N_DONORS, n_genes=N_GENES, min_n_cells=MIN_N_CELLS, max_n_cells=MAX_N_CELLS
):
    """Simulate an AnnData object with multiple donors.

    AnnData object with n_obs × n_vars = 445 × N_GENES
    obs: 'cell_label', 'donor_id'
    var: 'chrom', 'start', 'end'

    Parameters
    ----------
    n_donors : int, optional
        Number of donors to simulate, by default N_DONORS
    n_genes : int, optional
        Number of genes to simulate, by default N_GENES
    min_n_cells : int, optional
        Minimum number of cells per donor, by default MIN_N_CELLS
    max_n_cells : int, optional
        Maximum number of cells per donor, by default MAX_N_CELLS

    Returns
    -------
    AnnData
        Simulated AnnData object
    """
    adatas = []
    has_all_celltypes = [True] * (n_donors - 1) + [False]
    cum_n_cells = 0
    for i in range(n_donors):
        n_cells = np.random.randint(min_n_cells, max_n_cells)
        has_all_celltypes = i != 0  # the first donor misses one celltype
        adatas.append(_sim_donor(cum_n_cells, n_cells, n_genes, has_all_celltypes))
        cum_n_cells += n_cells
    donors = [f"D{i}" for i in range(n_donors)]
    adata = ad.concat(adatas, merge="first", keys=donors, label=DAnn.donor)
    return adata


def sim_gdata(n_donors=N_DONORS, n_snps=N_SNPS, adata=None):
    if adata is None:
        pos = np.arange(n_snps)
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
        index="SNP" + pd.RangeIndex(n_snps).astype(str),
    )
    gdata = ad.AnnData(X=X, obs=obs, var=var)
    return gdata
