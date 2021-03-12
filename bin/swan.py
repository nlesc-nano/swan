"""Statistical models."""
import argparse
import logging
import pickle
import tempfile
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from flamingo.features.featurizer import generate_fingerprints
from flamingo.log_config import configure_logger
from flamingo.utils import Options
from rdkit.Chem import AllChem, PandasTools
from sklearn.preprocessing import RobustScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .datasets import FingerprintsDataset, MolGraphDataset
from .input_validation import validate_input
from .load_models import select_model
from .plot import create_scatter_plot
from .geometry import read_geometries_from_files
from .early_stopping import EarlyStopping

__all__ = ["FingerprintModeller", "GraphModeller", "Modeller"]

# Starting logger
LOGGER = logging.getLogger(__name__)


def main():
    """Parse the command line arguments and call the modeller class."""
    parser = argparse.ArgumentParser("modeller")
    # configure logger
    parser.add_argument('-i', required=True, help="Input file with options")
    parser.add_argument("-m",
                        "--mode",
                        help="Operation mode: train or predict",
                        choices=["train", "predict"],
                        default="train")
    parser.add_argument("--restart",
                        help="restart training",
                        action="store_true",
                        default=False)
    parser.add_argument('-w', help="workdir", default=".")
    args = parser.parse_args()

    # start logger
    configure_logger(Path(args.w), "swan")

    # log date
    LOGGER.info(f"Starting at: {datetime.now()}")

    # Check that the input is correct
    opts = validate_input(Path(args.i))
    opts.mode = args.mode

    if args.mode == "train":
        opts.restart = args.restart
        train_and_validate_model(opts)
    else:
        predict_properties(opts)


def train_and_validate_model(opts: Options) -> None:
    """Train the model usign the data specificied by the user."""
    modeller = FingerprintModeller if 'fingerprint' in opts.featurizer else GraphModeller
    researcher = modeller(opts)
    researcher.scale_labels()
    researcher.split_data()
    researcher.load_data()
    researcher.train_model()
    predicted, expected = tuple(
        researcher.to_numpy_detached(x) for x in researcher.validate_model())
    if opts.scale_labels:
        predicted = researcher.transformer.inverse_transform(predicted)
        expected = researcher.transformer.inverse_transform(expected)
    create_scatter_plot(predicted, expected, opts.properties)


def predict_properties(opts: Options) -> pd.DataFrame:
    """Use a previous trained model to predict properties."""
    LOGGER.info(f"Loading previously trained model from: {opts.model_path}")
    modeller = FingerprintModeller if 'fingerprint' in opts.featurizer else GraphModeller
    researcher = modeller(opts)
    # Generate features
    if 'fingerprint' in opts.featurizer:
        features = generate_fingerprints(researcher.data['molecules'],
                                         opts.featurizer.fingerprint,
                                         opts.featurizer.nbits)
        features = torch.from_numpy(features).to(researcher.device)
    else:
        # Create a single minibatch with the data to predict
        dataset = MolGraphDataset(tempfile.mkdtemp(prefix="dataset_"),
                                  researcher.data)
        data_loader = tg.data.DataLoader(dataset=dataset,
                                         batch_size=len(
                                             researcher.data['molecules']))
        features = next(iter(data_loader))
        features.to(researcher.device)

    # Predict the property value and report
    predicted = researcher.to_numpy_detached(researcher.predict(features))
    # Transform back the labels

    # Load and applying the scales
    if opts.scale_labels:
        researcher.load_scale()
        transformed = researcher.transformer.inverse_transform(
            predicted).flatten()

    df = pd.DataFrame({
        'smiles': researcher.data['smiles'].to_numpy(),
        'predicted_property': transformed
    })
    path = Path(opts.workdir) / "prediction.csv"
    print("prediction data has been written to: ", path)
    df.to_csv(path, index=False)
    return df
