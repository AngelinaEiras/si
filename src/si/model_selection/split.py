import numpy as np
import random as rd
import typing
from si.data.dataset import Dataset


def train_test_split(dset : Dataset, test_size : float, random_state):
    rd.seed(random_state)
    n_samples = dset.shape()[0]
    permutacoes = np.random.permutation(n_samples)
    corte = int(n_samples*test_size)
    sample_index_train = permutacoes[corte:]
    sample_index_test = permutacoes[:corte]
    
    df = dset.to_dataframe
    dt_train = Dataset(df.iloc[sample_index_train], features=dset.features, label=dset.label)
    dt_test = Dataset(df.iloc[sample_index_test], features=dset.features, label=dset.label)
    return (dt_train, dt_test)

'''

def train_test_split(dataset, test_size, random_state):
    np.random.seed(random_state)

    n_samples= dataset.shape()[0]

    n_test= int(n_samples * test_size)

    permutations = np.random.permutation(n_samples)

    test_indx = permutations[:n_test]

    train_indx = permutations[n_test:]


    train = Dataset(dataset.X[train_indx], dataset.Y[train_indx], features = dataset.Features, label=dataset.Label)
    test = Dataset(dataset.X[test_indx], dataset.Y[test_indx], features = dataset.Features, label=dataset.Label)

    return train, test
'''


if __name__ == '__main__':
    import si.io.csv as CSV
    temp = CSV.read_csv('D:/Mestrado/2ano/1semestre/SIB/si/datasets/iris.csv', ',', True)

    x=np.array([[np.nan,1,3], [3,2,3], [3,np.nan,3]])
    y=np.array([1,2,5])
    features= ['A', 'B','C']        
    label= 'y'
    dataset= Dataset(X=x, y=None, features= features, label=None)
    # print(dataset.get_var())
    print(temp.fill_Na(0))
    print(train_test_split(dataset, 0.2, 5))