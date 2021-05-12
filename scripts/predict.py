#!/usr/bin/env python
from pathlib import Path

import pandas as pd
import torch

from swan.dataset import (DGLGraphData, FingerprintsData,
                          TorchGeometricGraphData)
from swan.modeller import Modeller
from swan.modeller.models import MPNN, FingerprintFullyConnected
from swan.modeller.models.se3_transformer import SE3Transformer
from swan.utils.plot import create_scatter_plot

path_files = Path("data/Carboxylic_acids/CDFT")
path_data = path_files / "cdft_random_500.csv"

properties = [
    "Dissocation energy (nucleofuge)",
    # "Dissociation energy (electrofuge)",
    # "Electroaccepting power(w+)",
    # "Electrodonating power (w-)",
    # "Electronegativity (chi=-mu)",
    # "Electronic chemical potential (mu)",
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
    path_data, properties=properties, sanitize=True)
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

# Predict data
torch.set_default_dtype(torch.float32)
researcher = Modeller(net, data, use_cuda=False)
researcher.load_model("swan_chk.pt")
predicted = researcher.predict(data.fingerprints)

# Scale the predicted data
data.load_scale()
print("labales: ", data.labels)
predicted = data.transformer.inverse_transform(predicted.numpy())

# Print the predicted vs the ground_true
ground_true = pd.read_csv(path_data, index_col=0)[properties].to_numpy()
create_scatter_plot(predicted, ground_true, properties)
df = pd.DataFrame({"expected": ground_true.flatten(), "predicted": predicted.flatten()})
df.to_csv("expected.csv")

