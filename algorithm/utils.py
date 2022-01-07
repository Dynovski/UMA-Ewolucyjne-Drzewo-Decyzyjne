import numpy as np

from sklearn import tree
from sklearn.metrics import confusion_matrix

import data_processing.data as dt


def get_basic_tree_classifier(data: dt.Data) -> tree.DecisionTreeClassifier:
    train, validate, test = data.train_validate_test_split()
    labels = train[data.class_column_name]
    data_df = train.loc[:, train.columns != data.class_column_name]
    clf = tree.DecisionTreeClassifier()
    return tree.DecisionTreeClassifier().fit(data_df, labels)


def get_confusion_matrix(clf: tree.DecisionTreeClassifier, data: dt.Data) -> np.ndarray:
    train, validate, test = data.train_validate_test_split()
    validate_class_data = list(validate[data.class_column_name])
    validate_arguments = validate.loc[:, validate.columns != data.class_column_name]
    predictions = list(clf.predict(validate_arguments))
    matrix = confusion_matrix(validate_class_data, predictions, labels=data.classes)
    return matrix

