
from pathlib import Path
from swan.state import StateH5
import numpy as np
import pytest
import pandas as pd

from .utils_test import PATH_TEST


def test_state(tmp_path: Path, capsys):
    """Check that the class behaves as expected."""
    path_hdf5 = tmp_path / "swan_state.h5"
    state = StateH5(path_hdf5)

    node = "data"
    data = np.random.normal(size=15).reshape(3, 5)
    state.store_array(node, data)
    assert state.has_data(node)

    tensor = state.retrieve_data(node)
    assert np.allclose(tensor, data)

    state.show()
    out, _ = capsys.readouterr()
    assert "Available data" in out


def test_state_unknown_key(tmp_path: Path):
    """Check that an error is raised if there is not data."""
    path_hdf5 = tmp_path / "swan_state.h5"
    state = StateH5(path_hdf5)

    with pytest.raises(KeyError):
        state.retrieve_data("nonexisting property")


def store_smiles_in_state(tmp_path: Path):
    """Check that the smiles are correctly stored in the HDF5."""
    path_hdf5 = tmp_path / "swan_state.h5"
    path_smiles = PATH_TEST / "smiles.csv"
    df = pd.read_csv(path_smiles)
    smiles = df.smiles.to_numpy()

    state = StateH5(path_hdf5)
    state.store_array("smiles", smiles, "str")
    data = [x.decode() for x in state.retrieve_data("smiles")]
    assert data == smiles.tolist()
