# Author: Zbigniew Dynowski
from sklearn.tree import DecisionTreeClassifier

from data_processing.utils import get_data_loader, DatasetType
from data_processing.dataloader import DataLoader
from algorithm.utils import cross_validate, cross_validate_parallel
from algorithm.edt import EvolutionaryDecisionTree


def test():
    dataloader = get_data_loader(DatasetType.BANK)
    data = dataloader.get_data()
    labels = dataloader.get_labels()
    dt = DecisionTreeClassifier()
    print(f'Basic tree classifier accuracy: '
          f'{cross_validate(dt, data, labels, encode=True) * 100:.3f}%')

    edt = EvolutionaryDecisionTree(DataLoader.attributes_info(data), labels.unique())
    print(f'Edt classifier accuracy: {cross_validate_parallel(edt, data, labels) * 100:.3f}%')


if __name__ == "__main__":
    test()
