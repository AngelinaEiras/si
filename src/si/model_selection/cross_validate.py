import numpy as np
from si.data.dataset import Dataset
from si.model_selection.split import train_test_split
from collections.abc import Callable


def cross_validate(model, dataset: Dataset, scoring: Callable = None, cv: int = 3, test_size: float = 0.2) -> dict:
    """_summary_

    Args:
        model (_type_): model to validate.
        dataset (Dataset): validation Dataset.
        scoring (Callable, optional): scoring function. Defaults to None.
        cv (int, optional): number of folds. Defaults to 3.
        test_size (float, optional): size of the test dataset. Defaults to 0.2.

    Returns:
        scores (dict): dict containing seeds used in each fold, list with the training scores, and another containing the corresponding test scores.
    """
    scores = {
        'seeds': [],
        'train': [],
        'test': []
    }

    # para cada validação
    for _ in range(cv):
        # get seed
        seed = np.random.randint(0, 1000)

        # guarda no dict
        scores['seeds'].append(seed)

        train, test = train_test_split(dataset, test_size, random_state=seed)

        # fazer fit do dataset de treino
        model.fit(train)

        if scoring is None:
            scores['train'].append(model.score(train))
            scores['test'].append(model.score(train))

        else:
            y_train = train.y
            y_test = test.y

            scores['train'].append(scoring(y_train, model.predict(train)))
            scores['test'].append(scoring(y_test, model.predict(train)))

    return scores
