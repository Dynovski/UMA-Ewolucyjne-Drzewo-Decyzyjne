import pandas as pd

from sklearn.model_selection import KFold


def cross_validate(model, data: pd.DataFrame, labels: pd.DataFrame, num_splits: int, num_repeats: int = 25) -> float:
    accuracy: float = 0.0
    for k in range(num_repeats):
        kf = KFold(num_splits, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(data)):
            train_data: pd.DataFrame = data.iloc[train_index]
            train_labels: pd.Series = labels.iloc[train_index]
            test_data: pd.DataFrame = data.iloc[test_index]
            test_labels: pd.Series = labels.iloc[test_index]
            model.fit(train_data, train_labels)
            score: float = model.score(test_data, test_labels)
            print(f'{k + 1}.{i + 1}: accuracy: {score * 100:.3f}%')
            accuracy += score
    return accuracy / (num_repeats * num_splits)
