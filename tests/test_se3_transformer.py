import numpy as np
import torch

from swan.dataset import DGLGraphData
from swan.dataset.dgl_graph_data import dgl_data_loader
from swan.modeller import TorchModeller
from swan.modeller.models import TFN, SE3Transformer

from .utils_test import PATH_TEST, remove_files

NUM_LAYERS = 2     # Number of equivariant layers
NUM_CHANNELS = 4  # Number of channels in middle layers


torch.set_default_dtype(torch.float32)

CSV_FILE = PATH_TEST / "thousand.csv"
DATA = DGLGraphData(CSV_FILE, properties=["gammas"])


def run_modeller(net: torch.nn.Module):
    """Run a given model."""
    modeller = TorchModeller(net, DATA, use_cuda=False, replace_state=False)

    modeller.data.scale_labels()
    modeller.train_model(nepoch=1, batch_size=64)
    expected, predicted = modeller.validate_model()
    err = torch.functional.F.mse_loss(expected, predicted)
    assert not np.isnan(err.item())
    remove_files()


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
    researcher = TorchModeller(net, DATA, use_cuda=False)
    researcher.load_model("swan_chk.pt")

    # Predict the properties
    graphs = DATA.molecular_graphs
    inp_data = dgl_data_loader(DATA.dataset, batch_size=len(graphs))
    item = next(iter(inp_data))[0]
    predicted = net(item)

    # Scale the predicted data
    DATA.load_scale()
    predicted = DATA.transformer.inverse_transform(predicted.detach().numpy())

    assert len(graphs) == len(predicted)
