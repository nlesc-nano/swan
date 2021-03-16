"""Miscellaneous plot functions."""
import os
from typing import Any, List, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

plt.switch_backend('agg')


def create_scatter_plot(
        predicted: np.ndarray, expected: np.ndarray, properties: List[str], workdir: str = ".") -> None:
    """Plot the predicted vs the expected values."""
    sns.set()

    # Dataframes with the results
    columns_predicted = [f"{p}_predicted" for p in properties]
    columns_expected = [f"{p}_expected" for p in properties]
    df_predicted = pd.DataFrame(predicted, columns=columns_predicted)
    df_expected = pd.DataFrame(expected, columns=columns_expected)
    data = pd.concat((df_predicted, df_expected), axis=1)

    # Number of features
    nfeatures = predicted.shape[1]

    # Create a subplot with at most 3 features per line
    rows = nfeatures // 3
    ncols = nfeatures if nfeatures < 3 else 3
    rows = rows if rows > 1 else 1
    fig, axis = plt.subplots(nrows=rows, ncols=ncols, figsize=(20, 20), constrained_layout=True)
    # fig.tight_layout()
    if rows == 1:
        axis = [axis]

    for row, labels in enumerate(chunks_of(list(zip(columns_expected, columns_predicted)), 3)):
        for col, (label_x, label_y) in enumerate(labels):
            ax = axis[row][col] if nfeatures > 1 else axis[0]
            sns.regplot(x=label_x, y=label_y, data=data, ax=ax)

    path = os.path.join(workdir, "scatterplot.png")
    plt.savefig(path)

    print(f"{'name':40} slope intercept rvalue stderr")
    for k, name in enumerate(properties):
        # Print linear regression result
        reg = stats.linregress(predicted[:, k], expected[:, k])
        print(f"{name:40} {reg.slope:.3f}  {reg.intercept:.3f}  {reg.rvalue:.3f}  {reg.stderr:.3e}")


def chunks_of(data: List[Any], n: int) -> Iterator[Any]:
    """Return chunks of ``n`` from ``data``."""
    for i in range(0, len(data), n):
        yield data[i:i + n]