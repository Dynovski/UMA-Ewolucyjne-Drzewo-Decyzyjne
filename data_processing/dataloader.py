import pandas as pd

from typing import List, Dict, Any


class DataLoader:
    def __init__(self, data_path: str, class_column_name: str, attributes: List[str], separator: str = ','):
        self.class_column_name: str = class_column_name
        self.attributes: List[str] = attributes
        self.data: pd.DataFrame = pd.read_csv(data_path, header=None, sep=separator)
        self.data.set_axis(attributes + [class_column_name], axis=1, inplace=True)
        self.classes: List[str] = list(self.data[self.class_column_name].unique())
        self.data.dropna(how='any', inplace=True)

    def get_data(self) -> pd.DataFrame:
        return self.data[self.attributes]

    def get_labels(self) -> pd.Series:
        return self.data[self.class_column_name].astype(str)

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


class IrisDataLoader(DataLoader):
    def __init__(self):
        super(IrisDataLoader, self).__init__(
            'datasets/Iris/iris.data',
            'Class',
            ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
        )


class AbaloneDataLoader(DataLoader):
    def __init__(self):
        super(AbaloneDataLoader, self).__init__(
            'datasets/Abalone/abalone.data',
            'Age',
            ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
             'Shucked weight', 'Viscera weight', 'Shell weight']
        )


class AdultDataLoader(DataLoader):
    def __init__(self):
        super(AdultDataLoader, self).__init__(
            'datasets/Adult/adult.data',
            'Yearly-income',
            ['Age', 'Workclass', 'Final-weight', 'Education', 'Education-num', 'Marital-status', 'Occupation',
             'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-country']
        )


class WineQualityRedDataLoader(DataLoader):
    def __init__(self):
        super(WineQualityRedDataLoader, self).__init__(
            'datasets/WineQuality/winequality-red.csv',
            'Quality',
            ['Fixed acidity', 'Volatile acidity', 'Citrid acid', 'Residual sugar',
             'Chlorides', 'Free sulfur dioxide', 'Total sulfur dioxide', 'Density', 'PH', 'Sulphates', 'Alcohol'],
            ';'
        )


class WineQualityWhiteDataLoader(DataLoader):
    def __init__(self):
        super(WineQualityWhiteDataLoader, self).__init__(
            'datasets/WineQuality/winequality-white.csv',
            'Quality',
            ['Fixed acidity', 'Volatile acidity', 'Citrid acid', 'Residual sugar',
             'Chlorides', 'Free sulfur dioxide', 'Total sulfur dioxide', 'Density', 'PH', 'Sulphates', 'Alcohol'],
            ';'
        )


class BankDataLoader(DataLoader):
    def __init__(self):
        super(BankDataLoader, self).__init__(
            'datasets/Bank/bank-full.csv',
            'Has-subscribed',
            ['Age', 'Job', 'Marital', 'Education', 'Default', 'Balance', 'Housing', 'Loan',
             'Contact', 'Day', 'Month', 'Duration', 'Campaign', 'Pdays', 'Previous', 'Poutcome'],
            ';'
        )
