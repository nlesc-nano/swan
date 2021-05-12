#!/usr/bin/env python

import logging
from pathlib import Path
import torch
from swan.dataset import TorchGeometricGraphData, FingerprintsData, DGLGraphData
from swan.modeller import Modeller
from swan.modeller.models import FingerprintFullyConnected, MPNN, InvariantPolynomial
from swan.modeller.models.se3_transformer import TFN, SE3Transformer
from swan.utils.log_config import configure_logger
from swan.utils.plot import create_scatter_plot

configure_logger(Path("."))

# Starting logger
LOGGER = logging.getLogger(__name__)


# Path to the DATASET
path_files = Path("tests/files")
path_data = path_files / "cdft_properties.csv"
path_geometries = path_files / "cdft_geometries.json"


# Training variables
nepoch = 150
batch_size = 64
properties = [
    # "Dissocation energy (nucleofuge)",
    # "Dissociation energy (electrofuge)",
    # "Electroaccepting power(w+)",
    # "Electrodonating power (w-)",
    # "Electronegativity (chi=-mu)",
    "Electronic chemical potential (mu)",
    # "Electronic chemical potential (mu+)",
    # "Electronic chemical potential (mu-)",
    # "Electrophilicity index (w=omega)",
    # "Global Dual Descriptor Deltaf+",
    # "Global Dual Descriptor Deltaf-",
    # "Hardness (eta)",
    # "Hyperhardness (gamma)",
    # "Net Electrophilicity",
    # "Softness (S)"
]
num_labels = len(properties)

# Datasets
data = FingerprintsData(
    path_data, properties=properties, sanitize=False)
# data = DGLGraphData(
#     path_data, properties=properties, file_geometries=path_geometries, sanitize=False)

# FullyConnected NN
net = FingerprintFullyConnected(hidden_cells=100, num_labels=num_labels)

# # Graph NN configuration
# net = MPNN(batch_size=batch_size, output_channels=40, num_labels=num_labels)

# # # e3nn Network
# # net = InvariantPolynomial(irreps_out=f"{num_labels}x0e")

# # se3 transformers
# num_layers = 2     # Number of equivariant layers
# num_channels = 8   # Number of channels in middle layers
# num_nlayers = 0    # Number of layers for nonlinearity
# num_degrees = 2    # Number of irreps {0,1,...,num_degrees-1}
# div = 4            # Low dimensional embedding fraction
# pooling = 'avg'    # Choose from avg or max
# n_heads = 1        # Number of attention heads

# net = SE3Transformer(
#     num_layers, num_channels, num_nlayers=num_nlayers, num_degrees=num_degrees, div=div,
#     pooling=pooling, n_heads=n_heads)

# training and validation
torch.set_default_dtype(torch.float32)
researcher = Modeller(net, data, use_cuda=False)
researcher.set_optimizer("Adam", lr=0.0005)
researcher.set_scheduler("StepLR", 0.1)
researcher.data.scale_labels()
researcher.train_model(nepoch=nepoch, batch_size=batch_size)
expected, predicted = [x.cpu().detach().numpy() for x in researcher.validate_model()]
create_scatter_plot(predicted, expected, properties)
