import numpy as np
import pandas as pd

from typing import List, Dict, Tuple, Any

import config as cfg


class Data:
    def __init__(self, data_path: str, class_column_name: str, attributes: List[str], separator: str = ','):
        self.class_column_name: str = class_column_name
        self.attributes: List[str] = attributes
        self.data: pd.DataFrame = pd.read_csv(data_path, header=None, sep=separator)
        self.data.set_axis(attributes + [class_column_name], axis=1, inplace=True)
        self.classes: List[str] = list(self.data[self.class_column_name].unique())
        self.attribute_info_d: Dict[str, Dict[str, Any]] = {}
        self._get_attributes_info()

    def train_test_split(self, train_ratio=cfg.TRAIN_RATIO, seed=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        np.random.seed(seed)
        permuted_indices = np.random.permutation(self.data.index)
        len_data: int = len(self.data.index)
        train_end_idx: int = int(train_ratio * len_data)
        train: pd.DataFrame = self.data.iloc[permuted_indices[:train_end_idx]]
        test: pd.DataFrame = self.data.iloc[permuted_indices[train_end_idx:]]
        return train, test

    def _get_attributes_info(self):
        for attribute in self.attributes:
            info: Dict[str, Any] = {}
            data_types = self.data.dtypes
            if data_types[attribute] == 'float64' or data_types[attribute] == 'int64':
                info['is_string'] = False
                info['min_value'] = self.data[attribute].min()
                info['max_value'] = self.data[attribute].max()
            else:
                info['is_string'] = True
                info['possible_values'] = self.data[attribute].unique()
            self.attribute_info_d[attribute] = info

    def drop_columns_in_place(self, columns: List[str]):
        self.data = self.data.drop(columns, axis=1)
        self.attributes = [attribute for attribute in self.attributes if attribute not in columns]
        for column in columns:
            self.attribute_info_d.pop(column, None)


class IrisData(Data):
    def __init__(self):
        super(IrisData, self).__init__(
            'data/Iris/iris.data',
            'Class',
            ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
        )


class AbaloneData(Data):
    def __init__(self):
        super(AbaloneData, self).__init__(
            'data/Abalone/abalone.data',
            'Rings',
            ['Sex', 'Lenght', 'Diameter', 'Height', 'Whole weight',
             'Shucked weight', 'Viscera weight', 'Shell weight']
        )


class AdultData(Data):
    def __init__(self):
        super(AdultData, self).__init__(
            'data/Adult/adult.data',
            'Class',
            ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation',
             'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-country']
        )


class WineData(Data):
    def __init__(self):
        super(WineData, self).__init__(
            'data/Wine/wine.data',
            'Class',
            ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
             'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
             'OD280/OD315 of diluted wines', 'Proline']
        )


class BankData(Data):
    def __init__(self):
        super(BankData, self).__init__(
            'data/Bank/bank-full.csv',
            'Class',
            ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
             'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'],
            ';'
        )
