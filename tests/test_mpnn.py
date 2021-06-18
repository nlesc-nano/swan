import unittest

import numpy as np
import torch
import torch_geometric

from swan.dataset import TorchGeometricGraphData
from swan.modeller import TorchModeller
from swan.modeller.models.graph_models import MPNN

from .utils_test import PATH_TEST, remove_files


class TestMPNN(unittest.TestCase):
    """Test the finger print models
    """
    def setUp(self):
        self.data = PATH_TEST / "thousand.csv"
        self.data = TorchGeometricGraphData(self.data, properties=["Hardness (eta)"])
        self.net = MPNN()
        self.modeller = TorchModeller(self.net, self.data, replace_state=True)

    def test_train_data_mpnn(self):

        self.modeller.data.scale_labels()
        self.modeller.train_model(nepoch=5, batch_size=64)
        expected, predicted = self.modeller.validate_model()
        assert not all(np.isnan(x).all() for x in (expected, predicted))
        remove_files()

    def test_predict(self):
        graphs = self.modeller.data.molecular_graphs
        inp_data = torch_geometric.data.DataLoader(graphs, batch_size=len(graphs))
        item, _ = self.data.get_item(next(iter(inp_data)))

        predicted = self.modeller.predict(item)
        assert len(graphs) == len(predicted)
