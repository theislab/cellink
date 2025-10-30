import torch
import numpy as np
import scanpy as sc
from typing import Literal, List, Optional, Tuple
from anndata.utils import asarray
from cellink._core.data_fields import DAnn
from cellink._core import DonorData
from typing import Tuple, Dict, Any

import logging
logger = logging.getLogger(__name__)

def run_mixmil(
    dd: DonorData,
    n_pcs: int = 50,
    donor_key: str = DAnn.donor,
    bag_phenotype_key: str = None,
    embedding_key: str = "X_pca",
    likelihood: Optional[Literal["binomial", "categorical"]] = "binomial",
    n_trials: Optional[int] = 2,
    n_epochs: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-3,
    encode_sex: bool = True,
    encode_age: bool = True,
    additional_covariates: Optional[List[str]] = None,
    dtype: str = "float32"
) -> Tuple[Dict[str, Any], "MixMIL"]:
    """
    Train a MixMIL model on donor-level data with flexible covariate encoding.

    Parameters
    ----------
    dd : DonorData
        DonorData object containing single-cell and donor-level information.
    n_pcs : int, default=50
        Number of principal components to compute if not already present.
    bag_phenotype_key : str
        Column name in dd.G.obs for bag-level phenotype.
    likelihood : {'binomial', 'categorical'}, optional
        Likelihood model for training.
    n_trials : int, optional
        Number of trials if using binomial likelihood.
    n_epochs : int, default=2000
        Number of training epochs.
    batch_size : int, default=64
        Training batch size.
    lr : float, default=1e-3
        Learning rate for optimizer.
    encode_sex : bool, default=True
        Whether to include sex as a covariate.
    encode_age : bool, default=True
        Whether to include age as a covariate.
    additional_covariates : list of str, optional
        List of additional column names in dd.G.obs or dd.G.obsm to include as covariates.

    Returns
    -------
    results : dict
        Training results from the MixMIL model.
    model : MixMIL
        Trained MixMIL model instance.
    """

    try:
        from mixmil import MixMIL
    except ImportError:
        raise ImportError("mixmil is required for `run_mixmil`. Install with `pip install cellink[mixmil]`.")

    if "X_pca" not in dd.C.obsm:
        logger.info("Calculating PCA.")
        sc.pp.pca(dd.C, n_comps=n_pcs)

    iidx = dd.C.obs[donor_key].astype("category").cat.codes
    indptr = np.concatenate([[0], np.cumsum(np.bincount(iidx))])
    X = dd.C.obsm[embedding_key].astype(dtype)
    Xs = [
        torch.from_numpy(X[start:end]) for start, end in zip(indptr[:-1], indptr[1:], strict=True)
    ]

    dtype = getattr(torch, dtype)
    covariate_list = []
    covariate_list.append(torch.ones((dd.shape[0], 1), dtype=dtype))

    if encode_sex:
        sex_tensor = torch.unsqueeze(torch.tensor(dd.G.obs["sex"].astype("category").cat.codes.values, dtype=dtype), dim=1)
        covariate_list.append(sex_tensor)

    if encode_age:
        age_tensor = torch.tensor(dd.G.obs[["age"]].values, dtype=dtype)
        mean = age_tensor.mean()
        std = age_tensor.std()
        tolerance = 1e-2 
        already_z_normalized = torch.isclose(mean, torch.tensor(0.0), atol=tolerance) and torch.isclose(std, torch.tensor(1.0), atol=tolerance)
        if not already_z_normalized and std > 0:
            logger.info("Performing z-normalization of age.")
            age_tensor = (age_tensor - mean) / std
        covariate_list.append(age_tensor)

    if additional_covariates:
        for cov in additional_covariates:
            if cov in dd.G.obs.columns:
                covariate = torch.from_numpy(dd.G.obs[[cov]].values.astype("float32"))
                covariate_list.append(covariate)
            elif cov in dd.G.obsm:
                covariate = torch.from_numpy(asarray(dd.G.obsm[cov]).astype("float32"))
                covariate_list.append(covariate)
            else:
                raise ValueError(f"Covariate '{cov}' not found in dd.G.obs or dd.G.obsm.")

    F = torch.cat(covariate_list, axis=1)

    if bag_phenotype_key in dd.G.var.index:
        Y = torch.tensor(asarray(dd.G.X[:, dd.G.var.index.get_indexer([bag_phenotype_key])]))
    else:
        Y = torch.unsqueeze(torch.tensor(dd.G.obs[bag_phenotype_key], dtype=torch.float32), dim=1)

    if likelihood is not None:
        model = MixMIL.init_with_mean_model(Xs, F, Y, likelihood=likelihood, n_trials=n_trials)
    else:
        Q = Xs[0].shape[1]
        K = F.shape[1]
        model = MixMIL(Q=Q, K=K)

    results = model.train(Xs, F, Y, n_epochs=n_epochs, batch_size=batch_size, lr=lr)

    return results, model