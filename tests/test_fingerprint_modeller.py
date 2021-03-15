from pathlib import Path
import unittest
import os
import numpy as np
import torch
from swan.modeller import FingerprintModeller
from swan.modeller.models.fingerprint_models import FingerprintFullyConnected

from swan.dataset import FingerprintsDataset

from .utils_test import PATH_TEST


class TestFingerprintModeller(unittest.TestCase):
    """Test the finger print models
    """
    def setUp(self):
        self.data = PATH_TEST / "thousand.csv"
        self.dataset = FingerprintsDataset(self.data, properties=["gammas"])
        self.net = FingerprintFullyConnected()
        self.modeller = FingerprintModeller(self.net, self.dataset)

    def test_train_data_fingerprints(self):

        self.modeller.scale_labels()
        self.modeller.create_data_loader(batch_size=64)
        self.modeller.train_model(nepoch=5)
        expected, predicted = self.modeller.validate_model()
        err = torch.functional.F.mse_loss(expected, predicted)
        assert not np.isnan(err.item())
