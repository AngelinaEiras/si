import numpy as np

class NN: # NeuroNetwork
    def __ini__(self, layers):
        self.layers= layers
    
    def fit(self, dataset):
        y = dataset.x
        for layer in self.layers:
            x = layer.forward(x)