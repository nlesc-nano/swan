from pathlib import Path

import numpy as np

from swan.utils.plot import create_scatter_plot


def test_plot(tmp_path: Path):
    """Check the plotting functionality."""
    dim = np.random.choice(np.arange(1, 10, dtype=int))
    xs = np.random.standard_cauchy(size=dim ** 2).reshape(dim, dim)
    ys = np.random.standard_normal(size=dim ** 2).reshape(dim, dim)
    properties = [f"property_{x}" for x in range(dim)]
    create_scatter_plot(xs, ys, properties, tmp_path)
