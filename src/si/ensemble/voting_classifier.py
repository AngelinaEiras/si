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
    def __init__(self, models: set()):
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
        return self
    
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
        mse = accuracy(dataset.y, self.predict(dataset))
        return mse

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.neighbors.knn_classifier import KNNClassifier
    from si.linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN and Logistic classifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    # initialize the Voting classifier
    voting = VotingClassifier([knn, lg])

    voting.fit(dataset_train)

    # compute the score
    score = voting.score(dataset_test)
    print(f"Score: {score}")

    print(voting.predict(dataset_test))
