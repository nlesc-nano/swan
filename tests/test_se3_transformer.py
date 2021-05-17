import numpy as np
import torch

from swan.dataset import DGLGraphData
from swan.dataset.dgl_graph_data import dgl_data_loader
from swan.modeller import Modeller
from swan.modeller.models import TFN, SE3Transformer

from .utils_test import PATH_TEST

NUM_LAYERS = 2     # Number of equivariant layers
NUM_CHANNELS = 4  # Number of channels in middle layers


torch.set_default_dtype(torch.float32)

CSV_FILE = PATH_TEST / "thousand.csv"
DATA = DGLGraphData(CSV_FILE, properties=["gammas"])


def run_modeller(net: torch.nn.Module):
    """Run a given model."""
    modeller = Modeller(net, DATA)

    modeller.data.scale_labels()
    modeller.train_model(nepoch=1, batch_size=64)
    expected, predicted = modeller.validate_model()
    err = torch.functional.F.mse_loss(expected, predicted)
    assert not np.isnan(err.item())


def test_tfn():
    """Check the TFN model."""
    net = TFN(NUM_LAYERS, NUM_CHANNELS)
    run_modeller(net)


def test_s3eTransformer_train():
    """Check the SE3 transformer model."""
    net = SE3Transformer(NUM_LAYERS, NUM_CHANNELS)
    run_modeller(net)


def test_se3Transformer_predict():
    """Check the prediction functionality of the SE3Transformer."""
    net = SE3Transformer(NUM_LAYERS, NUM_CHANNELS)
    researcher = Modeller(net, DATA, use_cuda=False)
    researcher.load_model("swan_chk.pt")

    # Predict the properties
    graphs = DATA.molecular_graphs
    inp_data = dgl_data_loader(DATA.dataset, batch_size=len(graphs))
    item = next(iter(inp_data))[0]
    predicted = net(item)
    assert len(graphs) == len(predicted)
