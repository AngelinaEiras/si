import numpy as np

class dense:
    def __init__(self, input_size:int, output_size:int):
        '''
        parametros
        nos = int

        nota: funçao pa ativaçao pa layer mas a layer pode ser uma funçao de ativaçao

        atributos
        weights
        bias
        '''
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(input_size, output_size) * 0.01 # inicializaçao da matriz de pesos com os valores q quisermos
        self.bias = np.zeros((1,output_size)) # inicializaçao com os valores q quisermos. 1 bias para cada output

        def forward(self, input_data: np.array):
            return np.dot(input_data, self.weights) + self.bias


class SigmoidActivation:
    def __init__(self):
        pass

    def forward(self, input_data):
        return 1/(1+np.exp(-input_data))
        # para fazer ativação, assumindo que é uma layer


'''
class Dense:

    def __init__(self, input_size, output_size):
        self.input_size = input_size  # Same size as attributes of input data
        self.output_size = output_size

        # Can be any initial values!
        self.weights = np.random.randn(input_size, output_size)  # (lines, columns) (normal distribution between 0 e 1)
        self.bias = np.zeros((1, self.output_size))

    def forward(self, input_data: np.array):  # Forward propagation
        """
        Parameters
        ----------
        :param input_data:
        :param X: Input data matrix (examples x attributes)
        """
        return np.dot(input_data,
                      self.weights) + self.bias  # Multiplpies input data lines (examples) with weights columns,
        # sums values of each column, and adds bias line


class SigmoidActivation:

    def __init__(self):
        pass

    def forward(self, input_data: np.array):
        return sigmoid_function(input_data)

    def backward(self, error: np.ndarray, learning_rate):

        return error


class SoftMaxActivation:
    def __init__(self):
        pass

    def forward(self, input_data: np.array):
        ez = np.exp(input_data)
        return ez / (np.sum(ez, axis=1, keepdims=True))


class ReLUActivation:

    def __init__(self):
        pass

    def forward(self, input_data: np.array):
        return np.maximum(input_data, 0)
'''
