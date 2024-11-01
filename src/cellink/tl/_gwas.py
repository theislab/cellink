import numpy as np
import scipy.linalg as la
import scipy.stats as st
import pandas as pd

from sklearn.preprocessing import quantile_transform
from time import time
from anndata import AnnData
from anndata.utils import asarray

class GWAS:
    r"""
    Linear model for univariate association testing
    between `P` phenotypes and `S` inputs (`P`x`S` tests)

    Parameters
    ----------
    Y : (`N`, `P`) ndarray
        outputs
    F : (`N`, `K`) ndarray
        covariates. If not specified, an intercept is assumed.
    """

    def __init__(self, Y, F=None):
        if F is None:
            F = np.ones((Y.shape[0], 1))
        self.Y = Y
        self.F = F
        self.df = Y.shape[0] - F.shape[1]
        self._fit_null()

    def _fit_null(self):
        """Internal functon. Fits the null model"""
        self.FY = np.dot(self.F.T, self.Y)
        self.FF = np.dot(self.F.T, self.F)
        self.YY = np.einsum("ip,ip->p", self.Y, self.Y)
        # calc beta_F0 and s20
        self.A0i = la.inv(self.FF)
        self.beta_F0 = np.dot(self.A0i, self.FY)
        self.s20 = (self.YY - np.einsum("kp,kp->p", self.FY, self.beta_F0)) / self.df

    def process(self, G, verbose=False):
        r"""
        Fit genotypes one-by-one.

        Parameters
        ----------
        G : (`N`, `S`) ndarray
            inputs
        verbose : bool
            verbose flag.
        """
        t0 = time()
        # precompute some stuff
        GY = np.dot(G.T, self.Y)
        GG = np.einsum("ij,ij->j", G, G)
        FG = np.dot(self.F.T, G)

        # Let us denote the inverse of Areml as
        # Ainv = [[A0i + m mt / n, m], [mT, n]]
        A0iFG = np.dot(self.A0i, FG)
        n = 1.0 / (GG - np.einsum("ij,ij->j", FG, A0iFG))
        M = -n * A0iFG
        self.beta_F = (
            self.beta_F0[:, None, :]
            + np.einsum("ks,sp->ksp", M, np.dot(M.T, self.FY)) / n[None, :, None]
        )
        self.beta_F += np.einsum("ks,sp->ksp", M, GY)
        self.beta_g = np.einsum("ks,kp->sp", M, self.FY)
        self.beta_g += n[:, None] * GY

        # sigma
        self.s2 = self.YY - np.einsum("kp,ksp->sp", self.FY, self.beta_F)
        self.s2 -= GY * self.beta_g
        self.s2 /= self.df

        # dlml and pvs
        self.lrt = -self.df * np.log(self.s2 / self.s20)
        self.pv = st.chi2(1).sf(self.lrt)

        t1 = time()
        if verbose:
            print("Tested for %d variants in %.2f s" % (G.shape[1], t1 - t0))

    def getPv(self):
        """
        Get pvalues

        Returns
        -------
        pv : ndarray
        """
        return self.pv

    def getBetaSNP(self):
        """
        get effect size SNPs

        Returns
        -------
        beta : ndarray
        """
        return self.beta_g

    def getLRT(self):
        """
        get lik ratio test statistics

        Returns
        -------
        lrt : ndarray
        """
        return self.lrt

    def getBetaSNPste(self):
        """
        get standard errors on betas

        Returns
        -------
        beta_ste : ndarray
        """
        beta = self.getBetaSNP()
        pv = self.getPv()
        z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
        ste = beta / z
        return ste

def _my_quantile_transform(x, seed=1):
    """
    Gaussian quantile transform for values in a pandas Series.    :param x: Input pandas Series.
    :type x: pd.Series
    :param seed: Random seed.
    :type seed: int
    :return: Transformed Series.
    :rtype: pd.Series    .. note::
        “nan” values are kept
    """
    np.random.seed(seed)
    x_transform = x.copy()
    if isinstance(x_transform, pd.Series):
        x_transform = x_transform.to_numpy()    
    is_nan = np.isnan(x_transform)
    n_quantiles = np.sum(~is_nan)    
    x_transform[~is_nan] = quantile_transform(
        x_transform[~is_nan].reshape([-1, 1]),
        n_quantiles=n_quantiles,
        subsample=n_quantiles,
        output_distribution="normal",
        copy=True,
    )[:, 0]    
    #x_transform = pd.Series(x_transform, index = x.index)
    return x_transform


def _store_fixed_effects(n_comps, pb_adata):
    logger.warning("Storing fixed effects")
    # compute expression PCs
    sc.pp.highly_variable_genes(pb_adata, n_top_genes=5000)
    sc.tl.pca(pb_adata, use_highly_variable=True, n_comps=n_comps)
    sc.tl.pca(pb_adata, n_comps=n_comps)
    pb_adata.obsm["E_dpc"] = column_normalize(pb_adata.obsm["X_pca"])
    # load patient covariates
    sex_one_hot = np.eye(2)[(pb_adata.obs.sex.values - 1)]
    age_standardized = StandardScaler().fit_transform(pb_adata.obs.age.values.reshape(-1, 1))
    pb_adata.obs["age_std"] = age_standardized
    pb_adata.obs[["sex1", "sex2"]] = sex_one_hot
    return pb_adata

def _run_gwas(pbdata: AnnData, gdata: AnnData, target_gene: str, cis_window: int):
    ## retrieving the pseudo-bulked data
    Y = pbdata[:, [target_gene]].layers["mean"]
    Y = asarray(Y)
    Y = _my_quantile_transform(Y)
    ## retrieving start and end position for each gene
    start = pbdata.var.loc[target_gene].start
    end = pbdata.var.loc[target_gene].end
    chrom = pbdata.var.loc[target_gene].chrom
    ## retrieving the variants within the cis window
    subgadata = gdata[:, (gdata.var.chrom == chrom) & (gdata.var.pos >= start - cis_window) & (gdata.var.pos <= end + cis_window)]
    G = subgadata.X.compute()
    gwas = GWAS(Y)
    gwas.process(G)
    return gwas, G.shape[1]

def get_best_eqtl_on_single_gene(pbdata: AnnData, gdata: AnnData, target_gene: str, cis_window: int):
    ## running gwas
    gwas, no_tested_variants = _run_gwas(pbdata, gdata, target_gene, cis_window)
    ## retrieve p-values
    pv = gwas.getPv()
    pv[np.isnan(pv)] = 1
    min_pv = pv.min()
    ## retrieving the variants associated with the lowest p-value
    min_pv_idx = pv.argmin()
    min_pv_variant = gdata.var.index[min_pv_idx]
    return {"target_gene": target_gene, "no_tested_variants": no_tested_variants, "min_pv": min_pv, "min_pv_variant": min_pv_variant}

def get_all_eqtls_on_single_gene(pbdata: AnnData, gdata: AnnData, target_gene: str, cis_window: int):
    ## running gwas
    gwas, no_tested_variants = _run_gwas(pbdata, gdata, target_gene, cis_window)
    ## retrieve p-values
    pv = gwas.getPv()
    pv[np.isnan(pv)] = 1
    ## retrieving beta and its std
    betasnp = gwas.getBetaSNP()
    betasnp_ste = gwas.getBetaSNPste()
    ## retrieving lrt 
    lrt = gwas.getLRT()
    ## retrieving the full results
    results = [
        {
            "target_gene": target_gene, 
            "no_tested_variants": no_tested_variants, 
            "pv": pv[idx], 
            "variant": gdata.var.index[idx],
            "betasnp": betasnp[idx],
            "betasnp_ste": betasnp_ste[idx],
            "lrt": lrt[idx]
        } 
        for idx in range(no_tested_variants)
    ]
    return results