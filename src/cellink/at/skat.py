import scipy
import numpy as np

def custom_getattr(name):
    if name in scipy.__dict__:
        return getattr(scipy, name)
    return getattr(np, name)

scipy.__getattr__ = custom_getattr
from gwas import GWAS
import scipy.stats as st


import scipy.linalg as la
import scipy.stats as st
from utils import davies_pvalue,xgower_factor_,skat_test

class Skat:
    def __init__(self, a=1, b=25, min_threshold=10, max_threshold=5000):
        """
        SKAT test for association between Y and X.
        Parameters
        ----------
        a : float (default=1)
            Parameter alpha of Beta distribution
        b : float  (default=25)
            Parameter beta of Beta distribution
        min_threshold : int (default=10)
            Minimum number of variants to perform the test 
        max_threshold : int (default=5000)
            Maximum number of variants to perform the test
        """
        assert a>0, 'Parameter alpha of Beta distribution must be > 0'
        assert b>0, 'Parameter beta of Beta distribution must be > 0'
        self.a = a
        self.b = b
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def run_skat(self, Y:np.array, X:np.array)-> float:
        """
        Method to perform SKAT test. Variants with a Minor Allele Count (MAC) <10 are collapsed together.
        If the number of variants is < 10, it returns NaN.
        It also returns NaN if the number of variants is > 5000.
        Same approach as Clarke,Holtkamp, et al. (https://doi.org/10.1038/s41588-024-01919-z)
        Parameters
        ----------
        Y : np.array
            Phenotype data
        X : np.array
            Genotype data
        a : float (default=1)
            Parameter alpha of Beta distribution
        b : float  (default=25)
            Parameter beta of Beta distribution
        min_threshold : int (default=10)
            Minimum number of variants to perform the test 
        max_threshold : int (default=5000)
            Maximum number of variants to perform the test
        
        Returns
        -------
        float
            p-value of the SKAT test 
        """
        Ilow = X.sum(0) < self.min_threshold
        if (~Ilow).sum() == 0:
            return np.nan
        else:
            _xlow = X[:, Ilow].sum(1)[:, None]
            _Xskat = np.concatenate([X[:, ~Ilow], _xlow], axis=1)
            if _Xskat.shape[1] > self.max_threshold:
                return np.nan
            else:
                maf = 0.5 * _Xskat.mean(0)
                maf[-1] = 0.5 * X[:, Ilow].mean()  # adjusts maf of aggregate burden
                _weights = st.beta.pdf(maf, self.a, self.b)
                _Xskat = (_Xskat - _Xskat.mean(0)) * np.sqrt(_weights)
                _Xskat = _Xskat / xgower_factor_(_Xskat)
        return skat_test(Y, _Xskat)


    