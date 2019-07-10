from swan.properties.properties import compute_mol_shape
import numpy as np


def test_mol_shape():
    """
    Test the calculation of a molecular shape as a grid
    """
    arr = compute_mol_shape("CO")

    assert ((arr.size == 64000) and np.all(np.isreal(arr)))
