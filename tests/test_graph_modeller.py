import numpy as np
import torch
from swan.modeller import Modeller
from swan.modeller.models.graph_models import MPNN
import unittest
from swan.dataset import GraphData

from .utils_test import PATH_TEST


class TestGraphModeller(unittest.TestCase):
    """Test the finger print models
    """
    def setUp(self):
        self.data = PATH_TEST / "thousand.csv"
        self.data = GraphData(self.data, properties=["gammas"])
        self.net = MPNN()
        self.modeller = Modeller(self.net, self.data)

    def test_train_data_fingerprints(self):

        self.modeller.data.scale_labels()
        self.modeller.train_model(nepoch=5, batch_size=64)
        expected, predicted = self.modeller.validate_model()
        err = torch.functional.F.mse_loss(expected, predicted)
        assert not np.isnan(err.item())
