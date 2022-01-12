# Author: Zbigniew Dynowski
from enum import Enum

import data_processing.dataloader as dl


class DatasetType(Enum):
    IRIS = 1
    ABALONE = 2
    WINE_RED = 3
    WINE_WHITE = 4
    ADULT = 5
    BANK = 6


def get_data_loader(dataset_type: DatasetType) -> dl.DataLoader:
    if dataset_type == DatasetType.IRIS:
        return dl.IrisDataLoader()
    elif dataset_type == DatasetType.ABALONE:
        return dl.AbaloneDataLoader()
    elif dataset_type == DatasetType.WINE_RED:
        return dl.WineQualityRedDataLoader()
    elif dataset_type == DatasetType.WINE_WHITE:
        return dl.WineQualityWhiteDataLoader()
    elif dataset_type == DatasetType.ADULT:
        return dl.AdultDataLoader()
    elif dataset_type == DatasetType.BANK:
        return dl.BankDataLoader()
