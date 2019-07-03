from swan.models.plot import create_scatter_plot
import numpy as np
import os


def test_scatterplot(tmp_path):
    """
    Check than an scatter plot is created
    """
    xs = np.random.normal(size=10)
    ys = np.random.normal(size=10)

    create_scatter_plot(xs, ys, workdir=tmp_path)

    assert os.path.exists(os.path.join(tmp_path, "scatterplot.png"))
