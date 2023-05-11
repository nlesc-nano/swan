import pytest

import numpy as np
import torch
import torch_geometric

from swan.dataset import TorchGeometricGraphData
from swan.modeller import TorchModeller
from swan.modeller.models.graph_models import MPNN

from tests.utils_test import PATH_TEST, remove_files

@pytest.fixture
def data():
    return TorchGeometricGraphData(PATH_TEST / "thousand.csv", properties=["Hardness (eta)"])

@pytest.fixture
def modeller(data):
    net = MPNN()
    return TorchModeller(net, data, replace_state=True)

def test_train_data_mpnn(modeller):
    modeller.data.scale_labels()
    modeller.train_model(nepoch=5, batch_size=64)
    expected, predicted = modeller.validate_model()
    assert not all(np.isnan(x).all() for x in (expected, predicted))
    remove_files()

def test_predict(modeller, data):
    graphs = modeller.data.molecular_graphs
    inp_data = torch_geometric.data.DataLoader(graphs, batch_size=len(graphs))
    item, _ = data.get_item(next(iter(inp_data)))

    predicted = modeller.predict(item)
    assert len(graphs) == len(predicted)
