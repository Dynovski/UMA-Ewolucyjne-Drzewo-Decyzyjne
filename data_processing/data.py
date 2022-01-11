import pandas as pd

from typing import List, Dict, Any


class Data:
    def __init__(self, data_path: str, class_column_name: str, attributes: List[str], separator: str = ','):
        self.class_column_name: str = class_column_name
        self.attributes: List[str] = attributes
        self.data: pd.DataFrame = pd.read_csv(data_path, header=None, sep=separator)
        self.data.set_axis(attributes + [class_column_name], axis=1, inplace=True)
        self.classes: List[str] = list(self.data[self.class_column_name].unique())

    def get_data(self) -> pd.DataFrame:
        return self.data[self.attributes]

    def get_labels(self) -> pd.DataFrame:
        return self.data[self.class_column_name]

    @staticmethod
    def attributes_info(data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        attribute_info_d: Dict[str, Dict[str, Any]] = {}
        for attribute in data.columns.tolist():
            info: Dict[str, Any] = {}
            data_types = data.dtypes
            if data_types[attribute] == 'float64' or data_types[attribute] == 'int64':
                info['is_string'] = False
                info['min_value'] = data[attribute].min()
                info['max_value'] = data[attribute].max()
            else:
                info['is_string'] = True
                info['possible_values'] = data[attribute].unique()
            attribute_info_d[attribute] = info
        return attribute_info_d


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
            'Class',
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
