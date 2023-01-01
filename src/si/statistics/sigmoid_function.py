import numpy as np

def sigmoid_function(x:np.ndarray)-> np.ndarray:
    """Sigmoid function"""
    return 1/(1 + np.exp(-x))
