import numpy as np
from si.metrics.accuracy import accuracy

class VotingClassifier:
    """
    
    Parameters
    ----------
    models: int
        conjunto de modelos.

    """
    def __init__(self, models: int):
        """
        Parameters
        ----------
        k: int
            Number of clusters.

        """
        # parameters
        self.models = models

    def fit(self, dataset):
        for model in self.models:
            model.fit(dataset)
        return self
    
    def predict(self, dataset):
        return
    
    def score():
        return

'''    def predict(self, dataset):

        def _get_majority_vote(pred:np.array) -> int:
            labels, counts = np.unique(pred, return_counts = True)
            return labels[np.argmax(counts)]

        vote = np.array([model.predict(dataset) for model in self.models]).transpose()
        return np.apply_along_axis(_get_majority_vote, axis = 1, arr = vote)

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
        y_pred_ = self.predict(dataset)
        return accuracy(dataset.Y, y_pred_)'''