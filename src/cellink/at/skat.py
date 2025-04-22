import scipy
import numpy as np

def custom_getattr(name):
    if name in scipy.__dict__:
        return getattr(scipy, name)
    return getattr(np, name)

scipy.__getattr__ = custom_getattr

import scipy.stats as st
from cellink.at.utils import xgower_factor_,skat_test
import pandas as pd
from typing import Union, List, Optional
import numpy as np
import pandas as pd
import anndata
from cellink._core import DonorData

#To handle multiple data types
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, List[float], List[int]]
DotPath = Union[str, List[str]]

DataContainer = Union[
    None,                 # If passing Y and X directly
    pd.DataFrame,         # If Y and X are in a DataFrame
    anndata.AnnData,      # If Y and X are in AnnData
    DonorData             # If Y and X are in DonorData
]

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
        Values for a, b are chosen accordingly to the SKAT original paper ( 10.1016/j.ajhg.2011.05.029)
        While values for min_threshold and max_threshold follow Clarke,Holthamp et al. (https://doi.org/10.1038/s41588-024-01919-z)
        """
        assert a>0, 'Parameter alpha of Beta distribution must be > 0'
        assert b>0, 'Parameter beta of Beta distribution must be > 0'
        self.a = a
        self.b = b
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def run_skat(self,
        data: DataContainer = None,
        Y: Optional[Union[ArrayLike, DotPath]] = None,
        X: Optional[Union[ArrayLike, DotPath]] = None
        ) -> float:
        if data is None:
            assert isinstance(Y,np.ndarray), "If data is None, Y must be provided and be a numpy array"
            assert isinstance(X,np.ndarray), "If data is None, X must be provided and be a numpy array"
            return self._run_skat(Y=Y, X=X)
        else:
            assert isinstance(data, (pd.DataFrame, anndata.AnnData, DonorData)), "data must be a pandas DataFrame, anndata.AnnData or DonorData"
            assert isinstance(Y, str), "Y must be a string or a list of strings"
            assert isinstance(X, (str,List[str])), "X must be a string or a list of strings" 
            if isinstance(X, str):
                X = [X]
            if isinstance(data, pd.DataFrame):
                assert Y in data.columns, "Y must be a column in the DataFrame."
                assert X in data.columns, "X must be a column in the DataFrame."
                Y = data[Y].values
                X = data[X].values
            elif isinstance(data, anndata.AnnData):
                assert Y in data.obs.columns, "Y must be a column in AnnData obs DataFrame."
                assert X in data.obs.columns, "X must be a column in AnnData obs DataFrame."
                Y = data.obs.loc[:, Y].values
                X = data.obs.loc[:, X].values
            elif isinstance(data, DonorData):
                assert Y in data.donor_data.obs.columns, "Y must be a column in the DonorData object."
                assert X in data.donor_data.obs.columns, "X must be a column in the DonorData object."
                Y = data.donor_data[Y].values
                X = data.donor_data[X].values
            return self._run_skat(Y=Y, X=X)

    def _run_skat(self, Y:np.ndarray=None, X:np.ndarray=None)-> float:
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


    