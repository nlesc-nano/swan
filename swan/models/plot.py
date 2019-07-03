import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_scatter_plot(predicted: np.ndarray, expected: np.ndarray) -> None:
    """
    Plot the predicted vs the expected values
    """
    predicted = predicted.flatten()
    expected = expected.flatten()
    sns.set()

    df = pd.DataFrame({'expected': expected, 'predicted': predicted})

    sns.scatterplot(x='expected', y='predicted', data=df)

    plt.savefig("scatterplot.png")
