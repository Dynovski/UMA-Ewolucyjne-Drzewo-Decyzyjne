import numpy as np

from sklearn.metrics import confusion_matrix
from typing import Any, List

from data_processing.data import Data


def get_confusion_matrix(clf: Any, data: Data) -> np.ndarray:
    test_labels = data.test_labels
    test_data = data.test_inputs
    predictions = list(clf.predict(test_data))
    matrix = confusion_matrix(test_labels, predictions, labels=data.classes)
    return matrix


def cross_validate(model, data: np.ndarray, labels: np.ndarray, num_splits: int, num_repeats: int = 25) -> float:
    accuracy: float = 0.0
    for k in range(num_repeats):
        indices = np.random.permutation(data.shape[0])
        shuffled_data: np.ndarray = data[indices]
        shuffled_labels: np.ndarray = labels[indices]
        data_splits: List[np.ndarray] = np.array_split(shuffled_data, num_splits)
        labels_splits: List[np.ndarray] = np.array_split(shuffled_labels, num_splits)
        for i in range(num_splits):
            train_data: np.ndarray = np.vstack(data_splits[:i] + data_splits[i + 1:])
            train_labels: np.ndarray = np.hstack(labels_splits[:i] + labels_splits[i + 1:])
            test_data: np.ndarray = data_splits[i]
            test_labels: np.ndarray = labels_splits[i]
            model.fit(train_data, train_labels)
            score: float = model.score(test_data, test_labels)
            print(f'{k}.{i}: accuracy: {score}')
            accuracy += score
    return accuracy / (num_repeats * num_splits)
