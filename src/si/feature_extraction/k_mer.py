import itertools
import numpy as np
from si.data.dataset import Dataset
from si.io.csv import read_csv


class KMer:
    """
    It performs k-means clustering on the dataset.
    It groups samples into k clusters by trying to minimize the distance between samples and their closest centroid.
    It returns the centroids and the indexes of the closest centroid for each point.

    Parameters
    ----------
    k: int
        string lenght.
    max_iter: int

    """

    def __init__(self, k: int = 3,  alphabet: str = 'DNA'):

        # parameters
        self.k = k
        if alphabet.upper() == 'DNA':
            self.alph = 'ACTG'
        elif alphabet.upper() == 'PROT':
            self.alph = 'ACDEFGHIKLMNPQRSTVWY'
        self.k_mers = None

    def fit(self, dataset: Dataset) -> 'KMer':
        """
        Determines all possible K-mers of a specified size ('k' parameter).
        Parameters
        ----------
        :param dataset: An instance of the Dataset class (required for consistency purposes only)
        """
        self.k_mers = ["".join(kmer) for kmer in itertools.product(
            self.alpha, repeat=self.k)]
        return self
    
    def _get_sequence_k_mer_composition(self, sequence: str) -> np.ndarray:
        """
        Calculates the k-mer composition of the sequence.
        Parameters
        ----------
        sequence : str
            The sequence to calculate the k-mer composition for.
        Returns
        -------
        list of float
            The k-mer composition of the sequence.
        """
        # calculate the k-mer composition
        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            counts[k_mer] += 1

        # normalize the counts
        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset.
        Parameters
        ----------
        dataset : Dataset
            The dataset to transform.
        Returns
        -------
        Dataset
            The transformed dataset.
        """
        # calculate the k-mer composition
        sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence)
                                       for sequence in dataset.X[:, 0]]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        # create a new dataset
        return Dataset(sequences_k_mer_composition, dataset.y, self.k_mers, dataset.label)

    def fit_transform(self, dataset: Dataset) -> 'Dataset':
        """
        Runs the fit() and transform() methods.
        Parameters
        ----------
        :param dataset: An instance of the Dataset class containing sequences
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    dataset_ = Dataset(X=np.array([['ACTGTTTAGCGGA', 'ACTGTTTAGCGGA']]),
                       y=np.array([1, 0]),
                       features=['sequence'],
                       label='label')

    k_mer_ = KMer(k=2)
    dataset_ = k_mer_.fit_transform(dataset_)
    print(dataset_.X)
    print(dataset_.features)

    print('---------------------------------------------------------')

    from sklearn.preprocessing import StandardScaler
    from si.io.csv import read_csv
    from si.linear_model.logistic_regression import LogisticRegression
    from si.model_selection.split import train_test_split
    import os
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../../../datasets/tfbs.csv')
    tfbs_dataset = read_csv(filename, sep=',', features=True, label=True)
    kmer = KMer(3)
    kmer_dataset = kmer.fit_transform(tfbs_dataset)
    print(kmer_dataset.features)

    kmer_dataset.X = StandardScaler().fit_transform(kmer_dataset.X)
    kmer_dataset_train, kmer_dataset_test = train_test_split(kmer_dataset, test_size=0.2)
    print(kmer_dataset.y)
    model = LogisticRegression()
    model.fit(kmer_dataset_train)
    score = model.score(kmer_dataset_test)
    print(f"Score: {score}")
