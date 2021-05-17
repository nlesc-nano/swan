#!/usr/bin/env python
from pathlib import Path

import argparse
from matplotlib.pyplot import sca
import pandas as pd
import torch
import torch_geometric as tg

from swan.dataset import (DGLGraphData, FingerprintsData,
                          TorchGeometricGraphData)
from swan.dataset.dgl_graph_data import dgl_data_loader
from swan.modeller import Modeller
from swan.modeller.models import MPNN, FingerprintFullyConnected
from swan.modeller.models.se3_transformer import SE3Transformer
from swan.utils.plot import create_scatter_plot
from collections import defaultdict

torch.set_default_dtype(torch.float32)

PATH_DATA = Path("data/Carboxylic_acids/CDFT/cdft_random_500.csv")

# Datasets
NUMLABELS = 1

PROPERTIES = [
    "Dissocation energy (nucleofuge)",
    "Dissociation energy (electrofuge)",
    "Electroaccepting power(w+)",
    "Electrodonating power (w-)",
    "Electronegativity (chi=-mu)",
    "Electronic chemical potential (mu)",
    "Electronic chemical potential (mu+)",
    "Electronic chemical potential (mu-)",
    "Electrophilicity index (w=omega)",
    "Global Dual Descriptor Deltaf+",
    "Global Dual Descriptor Deltaf-",
    "Hardness (eta)",
    "Hyperhardness (gamma)",
    "Net Electrophilicity",
    "Softness (S)"
]


def predict_fingerprints(path_parameters: Path, path_scales: Path):
    """Predict data using a previously trained fingerprint model."""
    data = FingerprintsData(PATH_DATA, sanitize=True)
    # FullyConnected NN
    net = FingerprintFullyConnected(hidden_cells=100, num_labels=NUMLABELS)
    return call_modeller(net, data, data.fingerprints, path_parameters, path_scales)


def predict_MPNN(path_parameters: Path, path_scales: Path):
    """Predict data using a previously trained MPNN model."""
    batch_size = 64
    output_channels = 40
    data = TorchGeometricGraphData(PATH_DATA, sanitize=True)

    # Graph NN configuration
    net = MPNN(batch_size=batch_size, output_channels=output_channels, num_labels=NUMLABELS)

    graphs = data.molecular_graphs
    inp_data = tg.data.DataLoader(graphs, batch_size=len(graphs))
    item, _ = data.get_item(next(iter(inp_data)))

    return call_modeller(net, data, item, path_parameters, path_scales)


def predict_SE3Transformer(path_parameters: Path, path_scales: Path):
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

    return call_modeller(net, data, item, path_parameters, path_scales)


def call_modeller(net, data, inp_data, path_parameters, path_scales):
    """Call the Modeller class to predict new data."""
    researcher = Modeller(net, data, use_cuda=False)
    researcher.load_model(path_parameters)
    predicted = researcher.predict(inp_data)

    # Scale the predicted data
    data.load_scale(path_scales)
    return data.transformer.inverse_transform(predicted.numpy())


def compare_prediction(predicted):
    "Print the predicted vs the ground_true."""
    ground_true = pd.read_csv(PATH_DATA, index_col=0)[PROPERTIES].to_numpy()
    create_scatter_plot(predicted, ground_true, PROPERTIES)
    df = pd.DataFrame({"expected": ground_true.flatten(), "predicted": predicted.flatten()})
    df.to_csv("expected.csv")


def compute_statistics(workdir: str, predictor):
    root = (workdir / "Results").absolute()
    ndirs = len(list(root.iterdir()))
    results = {name: [] for name in PROPERTIES}
    for name in PROPERTIES:
        for i in range(1, ndirs + 1):
            path = root / f"{i}" / name
            print("processing: ", path)
            parameters = next(path.glob("swan_chk.pt"))
            scales = next(path.glob("swan_scales.pkl"))
            results[name].append(predictor(parameters, scales))
    print(len(results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workdir", default=Path("."), type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--fingerprint", action="store_true")
    group.add_argument("-m", "--mpnn", action="store_true")
    group.add_argument("-s", "--se3transformer", action="store_true")
    args = parser.parse_args()
    if args.fingerprint:
        predictor = predict_fingerprints
    elif args.mpnn:
        predictor = predict_MPNN
    else:
        predictor = predict_SE3Transformer

    compute_statistics(args.workdir, predictor)


if __name__ == "__main__":
    main()
