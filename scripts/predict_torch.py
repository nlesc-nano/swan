#!/usr/bin/env python
from pathlib import Path

import pandas as pd
import torch
import torch_geometric as tg

from swan.dataset import (DGLGraphData, FingerprintsData,
                          TorchGeometricGraphData)
from swan.dataset.dgl_graph_data import dgl_data_loader
from swan.modeller import TorchModeller as Modeller
from swan.modeller.models import MPNN, FingerprintFullyConnected
from swan.modeller.models.se3_transformer import SE3Transformer
from swan.utils.plot import create_scatter_plot

torch.set_default_dtype(torch.float32)

path_files = Path("data/Carboxylic_acids/CDFT")
PATH_DATA = path_files / "cdft_random_500.csv"

# Datasets
NUMLABELS = 1


def predict_fingerprints():
    """Predict data using a previously trained fingerprint model."""
    data = FingerprintsData(PATH_DATA, sanitize=True)
    # FullyConnected NN
    net = FingerprintFullyConnected(hidden_units=100, output_units=NUMLABELS)
    return call_modeller(net, data, data.fingerprints)


def predict_MPNN():
    """Predict data using a previously trained MPNN model."""
    batch_size = 64
    output_channels = 40
    data = TorchGeometricGraphData(PATH_DATA, sanitize=True)

    # Graph NN configuration
    net = MPNN(batch_size=batch_size, output_channels=output_channels, num_labels=NUMLABELS)

    graphs = data.molecular_graphs
    inp_data = tg.data.DataLoader(graphs, batch_size=len(graphs))
    item, _ = data.get_item(next(iter(inp_data)))

    return call_modeller(net, data, item)


def predict_SE3Transformer():
    # se3 transformers
    num_layers = 2     # Number of equivariant layers
    num_channels = 8   # Number of channels in middle layers
    num_nlayers = 0    # Number of layers for nonlinearity
    num_degrees = 2    # Number of irreps {0,1,...,num_degrees-1}
    div = 4            # Low dimensional embedding fraction
    pooling = 'avg'    # Choose from avg or max
    n_heads = 1        # Number of attention heads

    net = SE3Transformer(
        num_layers, num_channels, num_nlayers=num_nlayers, num_degrees=num_degrees, div=div,
        pooling=pooling, n_heads=n_heads)

    data = DGLGraphData(PATH_DATA, sanitize=True)

    graphs = data.molecular_graphs
    inp_data = dgl_data_loader(data.dataset, batch_size=len(graphs))
    item = next(iter(inp_data))[0]

    return call_modeller(net, data, item)


def call_modeller(net, data, inp_data):
    """Call the Modeller class to predict new data."""
    researcher = Modeller(net, data, use_cuda=False)
    researcher.load_model("swan_chk.pt")
    predicted = researcher.predict(inp_data)

    # Scale the predicted data
    data.load_scale()
    return data.transformer.inverse_transform(predicted.numpy())


def compare_prediction(predicted):
    "Print the predicted vs the ground_true."""
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

    ground_true = pd.read_csv(PATH_DATA, index_col=0)[properties].to_numpy()
    create_scatter_plot(predicted, ground_true, properties)
    df = pd.DataFrame({"expected": ground_true.flatten(), "predicted": predicted.flatten()})
    df.to_csv("expected.csv")


def main():
    predicted = predict_fingerprints()
    # predicted = predict_MPNN()
    # predicted = predict_SE3Transformer()
    print(predicted)


if __name__ == "__main__":
    main()
