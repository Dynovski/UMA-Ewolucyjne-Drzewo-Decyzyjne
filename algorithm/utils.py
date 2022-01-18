# Author: Zbigniew Dynowski
import pandas as pd
import numpy as np

from multiprocessing import Pool
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef
from random import randrange, choice, uniform
from typing import Dict, Tuple, Any, List, Optional, Union

import config as cfg
from config import ALPHA, BETA, EXPECTED_TREE_HEIGHT, NUM_SPLITS, NUM_REPEATS


def cross_validate(model, data: pd.DataFrame, labels: pd.Series, num_splits: int = NUM_SPLITS,
                   num_repeats: int = NUM_REPEATS, encode: bool = False) -> Tuple[float, float, float, float]:
    accuracies: List[float] = []
    if encode:
        data = pd.get_dummies(data)
    for k in range(num_repeats):
        kf = KFold(num_splits, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(data)):
            train_data: pd.DataFrame = data.iloc[train_index]
            train_labels: pd.Series = labels.iloc[train_index]
            test_data: pd.DataFrame = data.iloc[test_index]
            test_labels: pd.Series = labels.iloc[test_index]
            model.fit(train_data, train_labels)
            accuracies.append(model.score(test_data, test_labels))
    return sum(accuracies) / (num_repeats * num_splits), np.std(accuracies).item(), max(accuracies), min(accuracies)


def cross_validate_parallel(model, data: pd.DataFrame, labels: pd.Series, num_splits: int = NUM_SPLITS,
                            num_repeats: int = NUM_REPEATS) -> Tuple[float, float, float, float]:
    accuracies: List[float] = []
    args_data: List[Tuple['EvolutionaryDecisionTree', pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]= []
    for k in range(num_repeats):
        kf = KFold(num_splits, shuffle=True)
        for train_index, test_index in kf.split(data):
            train_data: pd.DataFrame = data.iloc[train_index]
            train_labels: pd.Series = labels.iloc[train_index]
            test_data: pd.DataFrame = data.iloc[test_index]
            test_labels: pd.Series = labels.iloc[test_index]
            args_data.append((model, train_data, train_labels, test_data, test_labels))
    with Pool() as pool:
        scores: List[float] = pool.starmap(_fit_and_eval, args_data)
        accuracies += scores
    return sum(accuracies) / (num_repeats * num_splits), np.std(accuracies).item(), max(accuracies), min(accuracies)


def _fit_and_eval(model, train_data: pd.DataFrame, train_labels: pd.Series,
                  test_data: pd.DataFrame, test_labels: pd.Series) -> float:
    model.fit(train_data, train_labels)
    score = model.score(test_data, test_labels)
    return score


def choose_node_split_params(attributes_info: Dict[str, Dict[str, Any]]) -> Tuple[str, int, Union[str, float]]:
    attributes: List[str] = list(attributes_info.keys())
    attribute_idx: int = randrange(len(attributes))
    attribute: str = attributes[attribute_idx]
    attr_data: Dict[str, Any] = attributes_info[attribute]
    threshold: Optional[Union[float, str]] = None
    if attr_data['is_string']:
        threshold: str = choice(attr_data['possible_values'])
    else:
        threshold: float = round(uniform(attr_data['min_value'], attr_data['max_value']), 3)
    return attribute, attribute_idx, threshold


def evaluate_candidates(population: List['CandidateTree'], data: pd.DataFrame, labels: pd.Series) -> None:
    for individual in population:
        error: float = ALPHA * (1 - accuracy_score(labels.to_list(), individual.predict(data)))
        # changed so that smaller trees are not penalized if they have good score
        height_penalty: float = BETA * individual.height() / EXPECTED_TREE_HEIGHT
        individual.fitness = round(error + height_penalty, 3)
