import numpy as np

from sklearn import tree
from sklearn.metrics import confusion_matrix

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

