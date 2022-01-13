# Author: Zbigniew Dynowski
import sys

from sklearn.tree import DecisionTreeClassifier

import config as cfg
from data_processing.utils import get_data_loader, DatasetType
from data_processing.dataloader import DataLoader
from algorithm.utils import cross_validate, cross_validate_parallel
from algorithm.edt import EvolutionaryDecisionTree


def test():
    for dataset_type in DatasetType:
        print(dataset_type)
        dataloader = get_data_loader(dataset_type)
        data = dataloader.get_data()
        labels = dataloader.get_labels()
        data_info = DataLoader.attributes_info(data)
        unique_labels = labels.unique()
        dt = DecisionTreeClassifier()
        print(f'Basic tree classifier accuracy: '
              f'{cross_validate(dt, data, labels, encode=True) * 100:.3f}%')

        for population_size in cfg.POPULATION_SIZE_LIST:
            for expected_height in cfg.EXPECTED_TREE_HEIGHT_LIST:
                for split_prob in cfg.P_SPLIT_LIST:
                    for mutation_prob in cfg.P_MUTATION_LIST:
                        for crossover_prob in cfg.P_CROSSOVER_LIST:
                            edt = EvolutionaryDecisionTree(
                                data_info,
                                unique_labels,
                                population_size,
                                split_prob,
                                expected_height,
                                mutation_prob,
                                crossover_prob
                            )
                            print(f'Edt classifier accuracy: {cross_validate_parallel(edt, data, labels) * 100:.3f}%')


if __name__ == "__main__":
    test()
