import numpy as np
import torch

from swan.dataset import DGLGraphData
from swan.modeller import Modeller
from swan.modeller.models import TFN, SE3Transformer

from .utils_test import PATH_TEST

NUM_LAYERS = 1     # Number of equivariant layers
NUM_CHANNELS = 1  # Number of channels in middle layers


def run_modeller(net: torch.nn.Module):
    """Run a given model."""
    torch.set_default_dtype(torch.float32)
    csv_file = PATH_TEST / "thousand.csv"
    data = DGLGraphData(csv_file, properties=["gammas"])
    modeller = Modeller(net, data)

    modeller.data.scale_labels()
    modeller.train_model(nepoch=1, batch_size=64)
    expected, predicted = modeller.validate_model()
    err = torch.functional.F.mse_loss(expected, predicted)
    assert not np.isnan(err.item())


def test_s3e_transformer():
    """Check the SE3 transformer model."""
    net = SE3Transformer(NUM_LAYERS, NUM_CHANNELS)
    run_modeller(net)


def test_tfn():
    """Check the TFN model."""
    net = TFN(NUM_LAYERS, NUM_CHANNELS)
    run_modeller(net)
