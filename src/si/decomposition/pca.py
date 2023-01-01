import numpy as np
from si.data.dataset import Dataset


class PCA:
    '''
    Principal Component Analysis (PCA)
    '''

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean = None
        self.cent_data = None
        self.comp_princ = None
        self.explained_variance = None

    def fit(self, dataset: Dataset):
        """Estimate the average, the components and the variance

        Args:
            dataset (Dataset): input dataset

        Returns:
            self (PCA): self
        """
        self.mean= np.mean(dataset.X, axis=0)

        self.cent_data = np.subtract(dataset.X, self.mean)

        _, S, V_t = np.linalg.svd(
            self.cent_data, full_matrices=False)  # X = U*S*VT

        self.comp_princ = V_t[:self.n_components]

        n = len(dataset.X[:, 0])
        EV = (S**2)/(n-1)-n
        self.explained_variance = EV[:self.n_components]
        return self

    def transform(self, dataset: Dataset ):
        V = self.fit(dataset).comp_princ.T # matriz transporta
        # SVD reduced
        x_reduced = np.dot(self.cent_data, V)
        return x_reduced