from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
# quem utilizar isto tem de fazer import da sua função, como a f_clas é usada por defeito,
# faz se o import para estar disponível. Para se usar a f_regression fazer import desta


class SelectPercentile:
    """
    Select features according to the k highest scores.
    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    k: int, default=10
    percentile: int, default=5
        Number of top features to select.
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float = 0.5):
        self.percentil = percentile # 50% por defeito
        self.score_func = score_func # f_classification ou f_regression
        self.F = None
        self.p = None
    
    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self
        
    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the features in the top percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the features in the top percentile.
        """
        # nº de features q corresponde ao percentil
        k = int(len(dataset) * self.percentil)
        # ordem crescente
        idxs = np.argsort(self.F)[-k:]
        features = np.array(dataset.features)[idxs]
        return range(len(features[np.percentile]))

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectKBest and transforms the dataset by selecting the features in the top percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the the features in the top percentile.
        """
        self.fit(dataset)
        return self.transform(dataset)
