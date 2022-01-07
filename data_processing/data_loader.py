from enum import Enum

import data_processing.data as data


class Dataset(Enum):
    ABALONE = 1
    ADULT = 2
    BANK = 3
    IRIS = 4
    WINE = 5


def load_data(data_set: Dataset) -> data.Data:
    if data_set == Dataset.ABALONE:
        return data.AbaloneData()
    elif data_set == Dataset.ADULT:
        return data.AdultData()
    elif data_set == Dataset.BANK:
        return data.BankData()
    elif data_set == Dataset.IRIS:
        return data.IrisData()
    elif data_set == Dataset.WINE:
        return data.WineData()
