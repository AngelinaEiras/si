import numpy as np
from si.data.dataset import Dataset

class PCA:
    '''
    Principal Component Analysis (PCA)
    '''

    def __init__(self, n_components:int):
        self.n_components= n_components

    def fit(self, dataset: Dataset):
        self.mean= np.mean(dataset.X, axis=0)

        self.cent_data = np.subtract(dataset.X, self.mean)

        U, S, V_t = np.linalg.svd(self.cent_data, full_matrices=False) # X = U*S*VT

        self.comp_princ = V_t[:self.n_components]

        n = len(dataset.X[:, 0])
        EV = (S**2)/(n-1)-n
        self.explained_variance = EV[:self.n_components]
        return self

    def transform(self, dataset: Dataset ):
        V = self.fit(dataset).comp_princ.T # matriz transporta
        # SVD reduced
        Xreduced = np.dot(self.cent_data, V)
        return Xreduced