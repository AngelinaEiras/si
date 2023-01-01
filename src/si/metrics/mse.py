import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the mean squared error of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    error: float
        The mean squared error of the model
    """

    # m = len(y_true)
    error = np.square(np.subtract(y_true,y_pred)).mean()
    return error

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the mean squared error of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    error: float
        The mean squared error of the model
    """

    # m = len(y_true)
    error = -2 * (y_true - y_pred) / (len(y_true) * 2)
    return error