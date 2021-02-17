"""Miscellaneous plot functions."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

plt.switch_backend('agg')


def create_scatter_plot(predicted: np.ndarray, expected: np.ndarray, workdir: str = ".") -> None:
    """Plot the predicted vs the expected values."""
    sns.set()

    df = pd.DataFrame({'expected': expected, 'predicted': predicted})

    sns.regplot(x='expected', y='predicted', data=df)

    path = os.path.join(workdir, "scatterplot.png")
    plt.savefig(path)

    # Print linear regression result
    reg = stats.linregress(predicted, expected)
    print(reg)
