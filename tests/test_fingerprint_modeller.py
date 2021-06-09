import unittest
import numpy as np
import torch
from swan.modeller import TorchModeller
from swan.modeller.models.fingerprint_models import FingerprintFullyConnected
from swan.dataset import FingerprintsData

from .utils_test import PATH_TEST, remove_files


class TestFingerprintModeller(unittest.TestCase):
    """Test the finger print models"""
    def setUp(self):
        data = FingerprintsData(PATH_TEST / "thousand.csv", properties=["gammas"])
        self.net = FingerprintFullyConnected()
        self.modeller = TorchModeller(self.net, data)

    def test_train(self):
        self.modeller.data.scale_labels()
        self.modeller.train_model(nepoch=5, batch_size=64)
        expected, predicted = self.modeller.validate_model()
        err = torch.functional.F.mse_loss(expected, predicted)
        assert not np.isnan(err.item())
        remove_files()

    def test_predict(self):
        fingerprints = self.modeller.data.fingerprints
        predicted = self.modeller.predict(fingerprints)
        self.modeller.data.load_scale()
        predicted = self.modeller.data.transformer.inverse_transform(predicted.detach().numpy())

        assert len(predicted) == fingerprints.shape[0]
