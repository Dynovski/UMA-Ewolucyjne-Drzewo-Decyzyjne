from sklearn.tree import DecisionTreeClassifier

from data_processing.data_loader import load_data, Dataset
from algorithm.utils import cross_validate
from algorithm.edt import EvolutionaryDecisionTree
from visualizer.visualizer import save_confusion_matrix


def test():
    data = load_data(Dataset.IRIS)
    clf = DecisionTreeClassifier()
    # matrix = get_confusion_matrix(clf, data)
    # save_confusion_matrix(matrix, Dataset.IRIS, "test")
    print(f'Basic tree classifier accuracy: {cross_validate(clf, data.get_data(), data.get_labels(), 5)}')

    edt = EvolutionaryDecisionTree()
    # edt.fit(data.train_inputs, data.train_labels)
    # matrix_edt = get_confusion_matrix(edt, data)
    # save_confusion_matrix(matrix_edt, Dataset.IRIS, "edt test")
    print(f'Edt classifier accuracy: {cross_validate(edt, data.get_data(), data.get_labels(), 5)}')


if __name__ == "__main__":
    test()
