import numpy as np


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -(np.sum(y_true*np.log(y_pred)))/len(y_true)


def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # divide por 0?
    # return - ((y_true / y_pred) - ((1-y_true) / (1-y_pred))) / len(y_true) # ? https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
    return -y_true / (len(y_true)*y_pred)

y = np.random.rand(100, 1).T
y2 = np.random.rand(100, 1)
print(cross_entropy_derivative(y,y2).shape)
