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
    
    def predict(self):
        return
    
    def score():
        return