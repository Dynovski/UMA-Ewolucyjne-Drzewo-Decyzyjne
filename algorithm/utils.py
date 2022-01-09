import numpy as np

from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

import data_processing.data as dt


def get_basic_trained_tree_classifier(data: dt.Data) -> tree.DecisionTreeClassifier:
    train, test = data.train_test_split()
    labels = train[data.class_column_name]
    data_df = train.loc[:, train.columns != data.class_column_name]
    return tree.DecisionTreeClassifier().fit(data_df, labels)


def get_confusion_matrix(clf: tree.DecisionTreeClassifier, data: dt.Data) -> np.ndarray:
    train, test = data.train_test_split()
    test_data = list(test[data.class_column_name])
    validate_arguments = test.loc[:, test.columns != data.class_column_name]
    predictions = list(clf.predict(validate_arguments))
    matrix = confusion_matrix(test_data, predictions, labels=data.classes)
    return matrix


def cross_validate(model, data: np.ndarray, labels: np.ndarray, num_splits: int, num_repeats: int = 25) -> float:
    accuracy: float = 0.0
    for _ in range(num_repeats):
        scores: np.ndarray = cross_val_score(model, data, labels, cv=num_splits, n_jobs=4)
        accuracy += scores.mean()
    return accuracy / num_repeats
