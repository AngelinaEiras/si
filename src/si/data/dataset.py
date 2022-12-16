from typing import Tuple, Sequence

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None):
        """
        Dataset represents a machine learning tabular dataset.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")

        if features is None:
            features = [str(i) for i in range(X.shape[1])]
        else:
            features = list(features)

        if y is not None and label is None:
            label = "y"

        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def __str__(self):
        if not (self.Features is None):
            r=f'X:{str(self.Features)[:]}\n--\n'
        else:
            r = 'X:\n--\n'
        for elem in self.X:
            r += str(elem)[:].replace(' ', '\t') +'\n'
        if not (self.Y is None):
            r+= f'\nY: {self.Label}\n--\n' + str(self.Y).replace(' ', '\t') +'\n'
        return r

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.y is None:
            raise ValueError("Dataset does not have a label")
        return np.unique(self.y)

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    def dropna (self):
        """
        Removes all the samples that have at least a "null" value (NaN).
        Returns
        -------
        pandas.DataFrame (self.X).dropna(axis=0).reset_index(drop=True)
        """
        return pd.DataFrame(self.X).dropna(axis=0).reset_index(drop=True)
    
    '''
    def drop_na(self):

        df=pd.DataFrame(self.X, columns= self.Features)

        if not (self.Y is None):
            df.insert(loc=len(df), column=self.Label, value=self.Y)

            dfn=df.dropna()
            dt=dfn.to_numpy()

            if dt is not None:
                self.Y=[]
                for elem in dt[0:, -1:]:
                    self.Y.append(float(elem))
                self.X = dt[0:,:-1]
        else:
            dfn = df.dropna()
            dt = dfn.to_numpy()
            self.Y=[]
            self.X = dt[0:, :]

        return Dataset(self.X, self.Y, self.Features, self.Label)
    '''

    def fillna(self, value: int):
        """
        Replaces "null" values (NaN) by another value given by the user.
        -------
        pandas.DataFrame (self.X).fillna(value)
        """
        return pd.DataFrame(self.X).fillna(value)

    '''
    def fill_Na(self, n_or_m):
        df = pd.DataFrame(self.X, columns=self.Features)
        if not (self.Y is None):
            df.insert(loc=len(df), column=self.Label, value=self.Y)

            fill_df=df.fillna(n_or_m)
            dt = fill_df.to_numpy()

            if dt is not None:
                self.Y = []
                for elem in dt[0:, -1:]:
                    self.Y.append(float(elem))

                self.X = dt[0:, :-1]
        else:
            dfn = df.dropna()
            dt = dfn.to_numpy()
            self.Y=[]
            self.X = dt[0:, :]


        return Dataset(self.X, self.Y, self.Features, self.Label)
    '''

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)


'''
if __name__ == '__main__':
    import si.io.CSV as CSV
    temp = CSV.read_csv('D:/Mestrado/2ano/1semestre/SIB/si/datasets/iris.csv', ',', True)

    x=np.array([[np.nan,1,3], [3,2,3], [3,np.nan,3]])
    y=np.array([1,2,5])
    features= ['A', 'B','C']        
    label= 'y'
    dataset= Dataset(X=x, y=None, features= features, label=None)
    # print(dataset.get_var())
    print(temp.fill_Na(0))
'''
