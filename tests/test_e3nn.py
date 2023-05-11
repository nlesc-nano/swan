"""Test the interface to the e3nn library."""

import numpy as np
import torch

from swan.dataset import TorchGeometricGraphData
from swan.modeller import TorchModeller
from swan.modeller.models import InvariantPolynomial

from tests.utils_test import PATH_TEST, remove_files


def test_e3nn_equivariant():
    """Check that the interface to E3NN is working correctly."""
    path_data = PATH_TEST / "thousand.csv"
    data = TorchGeometricGraphData(path_data, properties=["Hardness (eta)"])
    net = InvariantPolynomial()
    modeller = TorchModeller(net, data, replace_state=True)
    modeller.set_optimizer('Adam', lr=0.001)
    modeller.set_scheduler("StepLR", 0.1)
    modeller.data.scale_labels()
    modeller.train_model(nepoch=5, batch_size=64)
    expected, predicted = modeller.validate_model()
    assert not all(np.isnan(x).all() for x in (expected, predicted))
    remove_files()
