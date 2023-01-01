import numpy as np
from si.statistics.sigmoid_function import sigmoid_function


class Dense:
    """
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer.
    Parameters
    ----------
    input_size: int
        The number of inputs the layer will receive.
    output_size: int
        The number of outputs the layer will produce.
    Attributes
    ----------
    weights: np.ndarray
        The weights of the layer.
    bias: np.ndarray
        The bias of the layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer.
        Parameters
        ----------
        input_size: int
            The number of inputs the layer will receive.
        output_size: int
            The number of outputs the layer will produce.
        """
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        self.X = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float = 0.001):
        """
        """
        error_for_backpropagation = np.dot(error, self.weights.T)
        self.weights -= learning_rate*np.dot(self.X.T, error)
        self.bias -= learning_rate*np.sum(error, axis=0)
        return error_for_backpropagation


class SigmoidActivation:
    """
    A sigmoid activation layer.
    """

    def __init__(self):
        """
        """
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        self.X = X
        return sigmoid_function(X)

    def backward(self, error: np.ndarray, learning_rate: float = "but why?..."):
        """
        Returns the input error value
        Parameters
        ----------
        error: np.ndarray
            The value of the error function derivative
        learning_rate: float
            The rate for the gradient descent
        """

        # https://stackoverflow.com/questions/28246231/about-backpropagation-and-sigmoid-function
        derivative = sigmoid_function(self.X) * (1 - sigmoid_function(self.X))
        error = error * derivative
        return error


class SoftMaxActivation:
    """
    A softmax activation layer.
    """

    def __init__(self):
        """
        """
        pass

    def forward(self, input_data: np.array):
        ez = np.exp(input_data)
        return ez / (np.sum(ez, axis=1, keepdims=True))


class ReLUActivation:
    """
    A ReLU activation layer
    """

    def __init__(self):
        """
        """
        self.X = None

    def forward(self, input_data: np.ndarray):
        self.X = input_data
        return np.maximum(0, input_data)

    def backward(self, error: np.ndarray, learning_rate: float = 0.001):
        self.X = np.where(self.X < 0, 0, 1)  # = 0 if x < 0 else 1
        error = error * self.X
        return error
