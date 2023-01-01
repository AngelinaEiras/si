
import numpy as np
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance
from si.data.dataset import Dataset


class KNNRegressor:
    """Estimate the mean value of the K nearest samples (regression)"""

    def __init__(self, k: int, distance=euclidean_distance):
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset: Dataset):
        """Store training dataset

        Args:
            dataset (Dataset): dataset to be stored

        Returns:
            self (KNNClassifier): self
        """
        self.dataset = dataset
        return self # no pdf mais abaixo diz "Resultado: dataset" <<--- ?

    def predict(self, dataset: Dataset):
        """Infer the Y classes for a given dataset

        Args:
            dataset (Dataset): test dataset

        Returns:
            predictions (np.ndarray): predictions for Y
        """
        predictions = np.apply_along_axis(
            self._get_closest_label, axis=1, arr=dataset.X)
        return predictions

    def _get_closest_label(self, sample: np.ndarray):
        """Calculate the label closest to a sample using the distance function

        Args:
            sample (np.ndarray): sample to be placed

        Returns:
            mean: label closest to the sample
        """
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        indexes = np.argsort(distances)[:self.k]

        # get the labels of the closet neighbors
        closest_y = self.dataset.y[indexes]

        # get mean
        mean = np.mean(closest_y)
        return mean

    def score(self, dataset: Dataset):
        """Calculate the error between real classes and predicted classes

        Args:
            dataset (Dataset): test dataset

        Returns:
            score (float): RMSE score of model predictions
        """
        score = rmse(dataset.y, self.predict(dataset))
        return score
