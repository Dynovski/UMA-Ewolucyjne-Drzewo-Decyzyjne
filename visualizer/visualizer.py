import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

import data_processing.data_loader as loader


def save_confusion_matrix(confusion_matrix: np.ndarray, data_type: loader.DatasetType, filename: str):
    save_dir: str = f'results/{data_type.name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = pd.DataFrame(confusion_matrix)
    sns.set(font_scale=1.4)
    sns.heatmap(df, annot=True, annot_kws={"size": 16})
    plt.savefig(f'{save_dir}/{filename}')
