from collections.abc import Callable
import itertools
from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate


def grid_search_cv(model, dataset: Dataset, parameter_grid: dict, scoring: Callable = None, cv: int = 3, test_size: float = 0.2) -> list(dict):
    for parameter in parameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(
                f'Modelo {model} não tem o parâmetro {parameter}')

    scores = []
    # cartesian product for all combinations
    for combination in itertools.product(*parameter_grid.values()):
        parameters = {}
        # update model parameters
        for parameter, value in zip(parameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value
        # perform cross-validation
        score = cross_validate(model, dataset, scoring, cv, test_size)
        # add parameters to cross-validation dict
        score.update({'parameters': parameters})
        # append to the result
        scores.append(score)
    return scores
