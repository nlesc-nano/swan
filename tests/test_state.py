
from pathlib import Path
from swan.state import StateH5
import numpy as np


def test_state(tmp_path: Path):
    """Check that the class behaves as expected."""
    path_hdf5 = tmp_path / "swan_state.h5"
    state = StateH5(path_hdf5)

    node = "data"
    data = np.random.normal(size=15).reshape(3, 5)
    state.store_array(node, data)
    assert state.has_data(node)

    tensor = state.retrieve_data(node)
    assert np.allclose(tensor, data)
