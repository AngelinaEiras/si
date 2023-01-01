from si.data.dataset import Dataset
import numpy as np
from si.metrics.accuracy import accuracy


class StackingClassifier:

    def __init__(self, models: list, final_mod):
        self.models = models  # lista de modelos já inicializados
        self.final_mod = final_mod

    def fit(self, dataset: Dataset):
        """Trains the intermediate models and the final model with the predictions of the former

        Args:
            dataset (Dataset): _description_

        Returns:
            self (StackingClassifier): self
        """

        # generate models for predictions
        for model in self.models:
            model.fit(dataset)

        # add predictions to dataset
        dataset_train = Dataset(dataset.X, dataset.y,
                                dataset.features, dataset.label)
        for model in self.models:
            dataset_train.X = np.c_[dataset_train, model.predict(dataset)]

        # train final_model
        self.final_model.fit(dataset_train)
        return self

    def predict(self, dataset: Dataset):
        """Generates predictions of the final model using the predictions of the previous models

        Args:
            dataset (Dataset): test dataset
        Returns:
            predictions: predictions of the final model
        """
        # add predictions to dataset
        dataset_final = Dataset(dataset.X, dataset.y, dataset.features, dataset.label)
        for model in self.models:
            dataset_final.X = np.c_[dataset_final.X, model.predict(dataset)]

        return self.final_model.predict(dataset_final)

    def score(self, dataset: Dataset):
        """Accuracy between real values and final model predictions

        Args:
            dataset (Dataset): test dataset

        Returns:
            score (float): accuracy of the final model
        """
        y_pred_ = self.predict(dataset)
        return accuracy(dataset.y, y_pred_)
