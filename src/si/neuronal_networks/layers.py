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

