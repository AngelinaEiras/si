from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function
from si.data.dataset import Dataset
#from si.metrics.mse import mse
matplotlib.use('TkAgg')


class LogisticRegression:

    def __init__(self, use_adaptive_alpha: bool = False, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000) -> None:

        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.use_adaptive_alpha = use_adaptive_alpha

        # attributes
        self.theta = None
        self.theta_zero = None
        self.history = {}
    
    def fit(self, dataset: Dataset) -> 'LogisticRegression':
        return self._regular_fit(dataset)

    def _regular_fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        Returns
        -------
        self: LogisticRegression
            The fitted model
        """

        m, n = dataset.shape()
        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = sigmoid_function(
                np.dot(dataset.X, self.theta) + self.theta_zero)

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * \
                np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term

            # custo
            custo = self.cost(dataset)
            if i == 0:
                self.history[i] = custo
            else:
                if np.abs(self.history.get(i - 1) - custo) >= 0.0001:
                    self.history[i] = custo
                else:
                    break
            self.theta_zero = self.theta_zero - \
                (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)
            # self.history[i] = custo

        return self

    def _adaptive_fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        Returns
        -------
        self: RidgeRegression
            The fitted model
        """

        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):

            # predicted y
            y_pred = sigmoid_function(
                np.dot(dataset.X, self.theta) + self.theta_zero)

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * \
                np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - \
                (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            # custo
            custo = self.cost(dataset)

            if i != 0:
                dif = abs(self.history.get(i - 1) - custo)

                if dif < 0.0001:
                    self.alpha = self.alpha / 2

            self.history[i] = custo

        no_dups = {}

        for key, value in self.history.items():
            if value not in no_dups.values():
                no_dups[key] = value

        self.history = no_dups

        return self

    def line_plot(self):
        iterations = list(self.history.keys())
        costs = list(self.history.values())
        plt.plot(iterations, costs)
        plt.show()

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of
        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """

        values = sigmoid_function(
            np.dot(dataset.X, self.theta) + self.theta_zero)
        arr = []

        for i in values:
            if i >= 0.5:
                arr.append(1)
            elif i < 0.5:
                arr.append(0)
        return np.array(arr)

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
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on
        Returns
        -------
        cost: float
            The cost function of the model
        """
        prediction = sigmoid_function(
            np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (-dataset.y * np.log(prediction)) - \
            ((1 - dataset.y) * np.log(1 - prediction))
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty * np.sum(self.theta **
                       2) / (2 * dataset.shape()[0]))
        return cost


if __name__ == '__main__':

    from si.model_selection.split import train_test_split
    from si.io.csv import read_csv
    import os
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../../../datasets/breast-bin.data')
    data1 = read_csv(filename, ",", False, True)

    # fit the model
    data1.X = StandardScaler().fit_transform(data1.X)
    train, test = train_test_split(data1, 0.3, 2)
    model = LogisticRegression()
    model._regular_fit(train)
    score = model.score(data1)
    cost = model.cost(data1)
    pred = model.predict(data1)
    model.line_plot()
    print(model.history)
    print(f"Parameters: {model.theta}")
    print(f"Score: {score}")
    print(f"Cost: {cost}")
    print(f"Predictions: {pred}")
    print(f"Score: {model.score(data1)}")
    print(f"Cost: {model.cost(data1)}")
