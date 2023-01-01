
import math
import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the squared root of the mean squared error of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    error: float
        The squared root of the mean squared error of the model
    """
    error = math.sqrt(np.square(np.subtract(y_true,y_pred)).mean())
    return error
