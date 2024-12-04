import numpy as np
from anndata import AnnData


def one_hot_encode_genotypes(
    adata: AnnData,
) -> AnnData | np.ndarray | None:
    """Convert genotypes to one-hot encoding.

    This function takes the genotype data in `adata` and converts it into a one-hot encoded format.
    Each genotype (e.g., homozygous reference, heterozygous, homozygous alternate) is transformed
    into a binary vector representation.

    Params
    ------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to individuals and columns to variants.

    Returns
    -------
    Returns the one-hot encoded genotypes or updates `adata` with the new representation, depending on `inplace`.

    Example
    --------
    >>> import numpy as np
    >>> from anndata import AnnData
    >>> adata = AnnData(
    ...     np.array(
    ...         [
    ...             [0, 1, 2],
    ...             [1, 0, 1],
    ...             [2, 2, 0],
    ...         ]
    ...     )
    ... )
    >>> one_hot_encoded = one_hot_encode_genotypes(adata, key_added="one_hot", inplace=False)
    >>> one_hot_encoded
    array([[[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]],

           [[0, 1, 0],
            [1, 0, 0],
            [0, 1, 0]],

           [[0, 0, 1],
            [0, 0, 1],
            [1, 0, 0]]])
    """
    genotypes = adata.X.astype(int)
    one_hot_encoded = np.zeros((genotypes.shape[0], genotypes.shape[1], 3), dtype=int)

    for i in range(genotypes.shape[0]):
        for j in range(genotypes.shape[1]):
            if genotypes[i, j] == 0:
                one_hot_encoded[i, j, 0] = 1  # Homozygous reference
            elif genotypes[i, j] == 1:
                one_hot_encoded[i, j, 1] = 1  # Heterozygous
            elif genotypes[i, j] == 2:
                one_hot_encoded[i, j, 2] = 1  # Homozygous alternate

    return one_hot_encoded


def dosage_per_strand(
    adata: AnnData,
) -> AnnData | np.ndarray | None:
    """Convert dosage to per-strand encoding.

    This function takes the dosage data in `adata` and converts it into a per-strand encoding format.
    If the dosage is 0, 1, or 2, it returns standard one-hot encoding. For other dosages, it returns
    the relative proportion of the dosage.

    Params
    ------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to individuals and columns to variants.

    Returns
    -------
    Returns the per-strand encoded genotypes or updates `adata` with the new representation, depending on `inplace`.

    Example
    --------
    >>> import numpy as np
    >>> from anndata import AnnData
    >>> adata = AnnData(
    ...     np.array(
    ...         [
    ...             [0, 1, 2],
    ...             [1, 1, 2],
    ...             [3, 4, 0],
    ...         ]
    ...     )
    ... )
    >>> per_strand_encoded = dosage_per_strand(adata, key_added="dosage", inplace=False)
    >>> per_strand_encoded
    array([[[1. , 0. ],
            [0.5, 0.5],
            [1. , 0. ]],

           [[0.5, 0.5],
            [0.5, 0.5],
            [1. , 0. ]],

           [[0.75, 0.25],
            [0.8, 0.2],
            [0. , 1. ]]])
    """
    dosages = adata.X.astype(float)
    per_strand_encoded = np.zeros((dosages.shape[0], dosages.shape[1], 2), dtype=float)

    for i in range(dosages.shape[0]):
        for j in range(dosages.shape[1]):
            if dosages[i, j] == 0:
                per_strand_encoded[i, j, 0] = 1.0  # Homozygous reference
                per_strand_encoded[i, j, 1] = 0.0
            elif dosages[i, j] == 1:
                per_strand_encoded[i, j, 0] = 0.5  # Heterozygous
                per_strand_encoded[i, j, 1] = 0.5
            elif dosages[i, j] == 2:
                per_strand_encoded[i, j, 0] = 0.0  # Homozygous alternate
                per_strand_encoded[i, j, 1] = 1.0
            else:
                per_strand_encoded[i, j, 0] = dosages[i, j] / (2)
                per_strand_encoded[i, j, 1] = 1 - per_strand_encoded[i, j, 0]

    return per_strand_encoded
