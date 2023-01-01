from si.statistics.euclidean_distance import euclidean_distance
import numpy as np
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset

class KNNClassifier:
    """Estimate the saple classes using KNN"""

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
        return self

    def predict(self, dataset: Dataset):
        """Infer the Y classes for a given dataset

        Args:
            dataset (Dataset): test dataset

        Returns:
            predictions (np.ndarray): predictions for Y
        """
        predictions = np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X) 
        return predictions

    def _get_closest_label(self, sample: np.ndarray):
        """Calculate the label closest to a sample using the distance function

        Args:
            sample (np.ndarray): sample to be placed

        Returns:
            closest_label: label closest to the sample
        """
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        indexes = np.argsort(distances)[:self.k]

        # get the labels of the closet neighbors
        closest_y = self.dataset.y[indexes]

        # get most common label
        labels, counts = np.unique(closest_y, return_counts=True)
        closest_label = labels[np.argmax(counts)]
        return closest_label

    def score(self, dataset: Dataset) -> float:
        """Calculate error between predictions and real values

        Args:
            dataset (Dataset): test dataset

        Returns:
            score (float): score
        """
        score = accuracy(dataset.y, self.predict(dataset))
        return score