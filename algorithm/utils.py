import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from random import randrange, choice, uniform
from typing import Dict, Tuple, Any, List, Optional, Union

from config import ALPHA, BETA, EXPECTED_TREE_HEIGHT


def cross_validate(model, data: pd.DataFrame, labels: pd.Series, num_splits: int = 5,
                   num_repeats: int = 1, encode: bool = False) -> float:
    accuracy: float = 0.0
    for k in range(num_repeats):
        kf = KFold(num_splits, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(data)):
            train_data: pd.DataFrame = data.iloc[train_index]
            train_labels: pd.Series = labels.iloc[train_index]
            test_data: pd.DataFrame = data.iloc[test_index]
            test_labels: pd.Series = labels.iloc[test_index]
            if encode:
                train_data = pd.get_dummies(train_data)
                test_data = pd.get_dummies(test_data)
            model.fit(train_data, train_labels)
            score: float = model.score(test_data, test_labels)
            print(f'{k + 1}.{i + 1}: accuracy: {score * 100:.3f}%')
            accuracy += score
    return accuracy / (num_repeats * num_splits)


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
        individual.fitness = error + height_penalty
