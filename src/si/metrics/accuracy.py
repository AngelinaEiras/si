import numpy as np

def accuracy(y_true: np, y_pred) -> float:
    """Calculate model accuracy from confusion matrix

    Args:
        y_true (np.ArrayLike): real Y values
        y_pred (np.ArrayLike): Y predictions

    Returns:
        error: model accuracy value
    """
    # erro = (VN + VP)/(VN+VP+FP+FN)
    error = np.sum(y_true == y_pred)/len(y_pred)
    return error
