import unittest

import numpy as np
import torch
import torch_geometric

from swan.dataset import TorchGeometricGraphData
from swan.modeller import Modeller
from swan.modeller.models.graph_models import MPNN

from .utils_test import PATH_TEST


class TestMPNN(unittest.TestCase):
    """Test the finger print models
    """
    def setUp(self):
        self.data = PATH_TEST / "thousand.csv"
        self.data = TorchGeometricGraphData(self.data, properties=["gammas"])
        self.net = MPNN()
        self.modeller = Modeller(self.net, self.data)

    def test_train_data_mpnn(self):

        self.modeller.data.scale_labels()
        self.modeller.train_model(nepoch=5, batch_size=64)
        expected, predicted = self.modeller.validate_model()
        err = torch.functional.F.mse_loss(expected, predicted)
        assert not np.isnan(err.item())

    def test_predict(self):
        graphs = self.modeller.data.molecular_graphs
        inp_data = torch_geometric.data.DataLoader(graphs, batch_size=len(graphs))
        item, _ = self.data.get_item(next(iter(inp_data)))

        predicted = self.modeller.predict(item)
        assert len(graphs) == len(predicted)