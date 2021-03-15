from pathlib import Path
import unittest
import os
import numpy as np
import torch
from swan.modeller import GraphModeller
from swan.modeller.models.graph_models import MPNN

from swan.dataset import MolGraphDataset

from .utils_test import PATH_TEST


class TestFingerprintModeller(unittest.TestCase):
    """Test the finger print models
    """
    def setUp(self):
        self.data = PATH_TEST / "thousand.csv"
        self.dataset = MolGraphDataset(self.data, properties=["gammas"])
        self.net = MPNN()
        self.modeller = GraphModeller(self.net, self.dataset)

    def test_train_data_fingerprints(self):

        self.modeller.scale_labels()
        self.modeller.split_data()
        self.modeller.load_data()
        self.modeller.train_model()
        expected, predicted = self.modeller.validate_model()
        err = torch.functional.F.mse_loss(expected, predicted)
        assert os.path.exists(self.modeller.opts.model_path)
        assert not np.isnan(err.item())
