from typing import Tuple, Union
import numpy as np
from scipy import stats
from si.data.dataset import Dataset


def f_regression(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray],
                                                Tuple[float, float]]:
    """
    Scoring function for regression problems.

    Parameters
    ----------
    dataset: Dataset
        A labeled dataset

    Returns
    -------
    F: np.array, shape (n_features,)
        F scores
    p: np.array, shape (n_features,)
        p-values
    """
    grauslib = len(dataset)-2
    pearson = [stats.pearsonr(i) for i in dataset]
    F = [((R**2)/(1-R**2))*grauslib for R in pearson]
    p = stats.f.sf(F,1,grauslib)
    return F, p
