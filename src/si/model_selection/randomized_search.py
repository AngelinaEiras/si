
from collections.abc import Callable
import numpy as np
from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate


def randomized_search_cv(model, dataset: Dataset, parameter_dist: dict, scoring: Callable = None, cv: int = 3, n_iter: int = 100, test_size: float = 0.2) -> list(dict):
    """Optimize model parameters using N random combinations

    Attributes:
        model (Any): Model to validate
        dataset (Dataset): Validation dataset
        parameter_dist (dict): Parametars to be used in the search
        scoring (Callable): Scoring function
        cv (int): Number of folds. Defaults to 3.
        n_iter (int): Number of random parameter combinations to be made. Defaults to 100.
        test_size (float): Size of the test dataset. Defaults to 0.2.
    
    Raises:
        AttributeError: Some model is missing parameters.

    Returns:
        scores (list(dict)): List of dictionaries containing parameter combinations used, seeds generated for each fold, plus its corresponding train and test results.
    """
    for parameter in parameter_dist:
        if not hasattr(model, parameter):
            raise AttributeError(
                f'Modelo {model} não tem o parâmetro {parameter}')

    scores = []

    for _ in range(n_iter):
        # get combinations
        parameters = {}
        for param in parameter_dist:
            value = np.random.choice(parameter_dist[param])
            setattr(model, param, value)
            parameters[param] = value
        # perform cross-validation
        score = cross_validate(model, dataset, scoring, cv, test_size)
        # add parameters to cross-validation dict
        score.update({'parameters': parameters})
        # append to the result
        scores.append(score)

    return scores
