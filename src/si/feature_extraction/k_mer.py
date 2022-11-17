from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance


class KMer:
    """
    It performs k-means clustering on the dataset.
    It groups samples into k clusters by trying to minimize the distance between samples and their closest centroid.
    It returns the centroids and the indexes of the closest centroid for each point.

    Parameters
    ----------
    k: int
        string lenght.
    max_iter: int

    """
    def __init__(self, k: int, seq: str):

        # parameters
        self.k = k
        self.seq = seq


    def fit(self):
        '''
        estima todos os k-mers possíveis; 
        retorna o self (ele próprio)
        '''
        self.k_mers = [''.join(k_mer) for k_mer in intertools.products]
        return self
    
    def transform(self):
        '''
        calcula a frequência normalizada de cada k-mer em cada sequência
        '''
        sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence)
                                        for sequence in dataset.X[:, 0]]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        return Dataset(X=sequences_k_mer_composition, y=dataset.y, features=self.k_mers, label=dataset.labels)

    def fit_transform(self):
        '''
        corre o fit e depois o transform
        '''
        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(seq)-self.k +1):
            k_mer = seq[i:i + self.k]
            counts[k_mer] +=1 
        
        #return np.array
        return self