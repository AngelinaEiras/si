from typing import Callable
import itertools
import numpy as np

from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from si.io.csv import read_csv


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
    def __init__(self, k: int,  alphabet = 'DNA'):

        # parameters
        self.k = k
        if alphabet.upper() == 'DNA':
            self.alph = 'ACTG'
        elif alphabet.upper() == 'PROT':
            self.alph = 'ACDEFGHIKLMNPQRSTVWY'
        self.k_mers = None


    def fit(self):
        '''
        estima todos os k-mers possíveis
        '''
        self.k_mers = [''.join(k_mer) for k_mer in itertools.products]
        return self
    
    def transform(self, dataset: Dataset):
        '''
        calcula a frequência normalizada de cada k-mer em cada sequência
        '''
        sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence) for sequence in dataset.X[:, 0]]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        return Dataset(X=sequences_k_mer_composition, y=dataset.y, features=self.k_mers, label=dataset.labels)  # create a new dataset

    def fit_transform(self):

        '''corre o fit e depois o transform'''

        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(seq)-self.k +1):
            k_mer = seq[i:i + self.k]
            counts[k_mer] +=1 
        
        #return np.array
        return self

if __name__ == '__main__':
    data1 = read_csv("/home/angelina/Desktop/2ano/si/datasets/tfbs.csv", ",", True, True)
    x = KMer(3)
    x.fit(data1)
    print(x.transform(data1))