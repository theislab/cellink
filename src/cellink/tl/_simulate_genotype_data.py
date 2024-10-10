import anndata
import anndata as ad
try:
    import msprime
except ImportError:
    print("msprime missing.")
import numpy as np


def simulate_genotype_data_msprime(
    n_individuals: int,
    n_variants: int,
    mutation_rate: float = 1e-8,
    recombination_rate: float = 1e-8,
) -> anndata.AnnData | None:
    """\
    Simulate genotype data using msprime.
    See https://tskit.dev/msprime/docs/stable/quickstart.html

    This function simulates genotype data for a given number of individuals and variants.

    Params
    ------
    n_individuals
        Number of individuals to simulate.
    n_variants
        Number of variants to simulate.
    mutation_rate
        The mutation rate for the simulation.
    recombination_rate
        The recombination rate for the simulation.

    Returns
    -------
    Returns the AnnData object with simulated genotype data or updates the input `adata` if inplace is True.

    Example
    --------
    >>> adata = anndata.AnnData(np.array([[0, 1], [1, 0], [0, 0]]))
    >>> simulate_genotype_data_msprime(adata, n_individuals=10, n_variants=5)
    """
    demography = msprime.Demography.isolated_model([n_individuals])

    ts = msprime.sim_ancestry(
        n_individuals, demography=demography, recombination_rate=recombination_rate, sequence_length=n_variants
    )

    mts = msprime.sim_mutations(ts, rate=mutation_rate)

    if mts.get_num_mutations() == 0:
        genotype_matrix = np.zeros((n_individuals, n_variants), dtype=int)
    else:
        genotype_matrix = np.array([variant.genotypes for variant in mts.variants()]).T

    # np.clip(genotype_matrix, 0, 2)

    if genotype_matrix.shape[1] < n_variants:
        genotype_matrix = np.pad(genotype_matrix, ((0, 0), (0, n_variants - genotype_matrix.shape[1])), mode="constant")

    adata = anndata.AnnData(genotype_matrix)
    adata.obs["individuals"] = np.arange(n_individuals)

    return adata


def simulate_genotype_data_numpy(
    n_individuals: int,
    n_variants: int,
) -> anndata.AnnData | None:
    """\
    Simulate random genotype data using numpy.

    This function generates a random genotype matrix with specified individuals and variants.

    Params
    ------
    n_individuals
        Number of individuals to simulate.
    n_variants
        Number of variants to simulate.

    Returns
    -------
    Returns the AnnData object with simulated genotype data or updates the input `adata` if inplace is True.

    Example
    --------
    >>> adata = anndata.AnnData(np.array([[0, 1], [1, 0], [0, 0]]))
    >>> simulate_genotype_data_numpy(adata, n_individuals=10, n_variants=5)
    """
    genotype_matrix = np.random.randint(0, 3, size=(n_individuals, n_variants))

    adata = ad.AnnData(genotype_matrix)
    adata.obs["individuals"] = np.arange(n_individuals)

    return adata
