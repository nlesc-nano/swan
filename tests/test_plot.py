from pathlib import Path

import numpy as np

from swan.utils.plot import create_scatter_plot, create_confidence_plot
from swan.modeller.gp_modeller import GPMultivariate


def remove_plot(file_name: str):
    p = Path(f"{file_name}.png")
    if p.exists():
        p.unlink()


def test_confidence_plot():
    """Check the confidence interval plotting functionality."""
    mean = np.random.normal(size=10)
    delta = 0.1 * np.random.normal(size=10)
    multi = GPMultivariate(mean, mean - delta, mean + delta)
    output_name = "confidence_plot"
    create_confidence_plot(multi, mean * 0.01, "awesomeness", output_name)
    remove_plot(output_name)


def test_scatterplot():
    """Check the plotting functionality."""
    dim = np.random.choice(np.arange(1, 10, dtype=int))
    xs = np.random.standard_cauchy(size=dim ** 2).reshape(dim, dim)
    ys = np.random.standard_normal(size=dim ** 2).reshape(dim, dim)
    properties = [f"property_{x}" for x in range(dim)]
    output_name = "scatter_plot"
    create_scatter_plot(xs, ys, properties, output_name)
    remove_plot(output_name)
