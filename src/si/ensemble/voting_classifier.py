import numpy as np
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset
Model = int

class VotingClassifier:
    """
    
    Parameters
    ----------
    models: int
        conjunto de modelos.

    """
    def __init__(self, models: set(Model)):
        """
        Parameters
        ----------
        models: set(Model)

        """
        # parameters
        self.models = models

    def fit(self, dataset):
        for model in self.models:
            model.fit(dataset)
    
    def predict(self, dataset):
        votes = np.array([model.predict(dataset) for model in self.models]).transpose()
        voted_values, vote_counts = np.unique(votes, return_counts = True)
        decision = voted_values[np.argmax(vote_counts)]
        return decision

    
    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on
        Returns
        -------
        accuracy: float
            The Mean Square Error of the model
        """
        return accuracy(dataset.Y, self.predict(dataset))
