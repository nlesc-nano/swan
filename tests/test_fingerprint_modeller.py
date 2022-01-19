import pytest
import numpy as np
import torch
from swan.modeller import TorchModeller
from swan.modeller.models.fingerprint_models import FingerprintFullyConnected
from swan.dataset import FingerprintsData

from .utils_test import PATH_TEST, remove_files


@pytest.fixture
def modeller():
    data = FingerprintsData(PATH_TEST / "thousand.csv", properties=["Hardness (eta)"])
    net = FingerprintFullyConnected()
    return TorchModeller(net, data)

def test_train(modeller):
    modeller.data.scale_labels()
    modeller.train_model(nepoch=5, batch_size=64)
    expected, predicted = modeller.validate_model()
    assert not all(np.isnan(x).all() for x in (expected, predicted))
    remove_files()

def test_predict(modeller):
    fingerprints = modeller.data.fingerprints
    predicted = modeller.predict(fingerprints)
    modeller.data.load_scale()
    predicted = modeller.data.transformer.inverse_transform(predicted.detach().numpy())

    assert len(predicted) == fingerprints.shape[0]
