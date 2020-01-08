"""Miscellaneous plot functions."""

import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def create_scatter_plot(predicted: np.ndarray, expected: np.ndarray, workdir: str = ".") -> None:
    """Plot the predicted vs the expected values."""
    predicted = predicted.flatten()
    expected = expected.flatten()
    sns.set()

    df = pd.DataFrame({'expected': expected, 'predicted': predicted})

    sns.regplot(x='expected', y='predicted', data=df)

    path = os.path.join(workdir, "scatterplot.png")
    plt.savefig(path)
