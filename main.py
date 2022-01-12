from sklearn.tree import DecisionTreeClassifier

from data_processing.utils import get_data_loader, DatasetType
from data_processing.dataloader import DataLoader
from algorithm.utils import cross_validate
from algorithm.edt import EvolutionaryDecisionTree


def test():
    dataloader = get_data_loader(DatasetType.IRIS)
    dt = DecisionTreeClassifier()
    print(f'Basic tree classifier accuracy: {cross_validate(dt, dataloader.get_data(), dataloader.get_labels())}')

    edt = EvolutionaryDecisionTree(DataLoader.attributes_info(dataloader.get_data()), dataloader.get_labels().unique())
    print(f'Edt classifier accuracy: {cross_validate(edt, dataloader.get_data(), dataloader.get_labels())}')


if __name__ == "__main__":
    test()
