# Author: Zbigniew Dynowski
from sklearn.tree import DecisionTreeClassifier

from data_processing.utils import get_data_loader, DatasetType
from data_processing.dataloader import DataLoader
from algorithm.utils import cross_validate
from algorithm.edt import EvolutionaryDecisionTree


def test():
    dataloader = get_data_loader(DatasetType.IRIS)
    dt = DecisionTreeClassifier()
    print(f'Basic tree classifier accuracy: '
          f'{cross_validate(dt, dataloader.get_data(), dataloader.get_labels(), encode=True) * 100:.3f}%')

    edt = EvolutionaryDecisionTree(DataLoader.attributes_info(dataloader.get_data()), dataloader.get_labels().unique())
    print(f'Edt classifier accuracy: {cross_validate(edt, dataloader.get_data(), dataloader.get_labels()) * 100 :.3f}%')


if __name__ == "__main__":
    test()
