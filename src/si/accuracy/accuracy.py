import numpy as np

def accuracy(y_true, y_pred):
   
    # VN - verd negativos;
    # VP - verd positivo;
    # FP - falso positivo;
    # FN falso neg.
    return np.sum(y_true == y_pred)/len(y_true) # erro = (VN + VP)/(VN+VP+FP+FN)