#!/usr/bin/env python

import logging
from pathlib import Path

import torch

from swan.dataset import FingerprintsData, split_dataset
from swan.modeller import GPModeller
from swan.modeller.models import GaussianProcess
from swan.utils.log_config import configure_logger
from swan.utils.plot import create_confidence_plot, create_scatter_plot

# Starting logger
configure_logger(Path("."))
LOGGER = logging.getLogger(__name__)

# Set float size default
torch.set_default_dtype(torch.float32)

# Path to the DATASET
path_data = Path("tests/files/thousand.csv")

# Training variables
nepoch = 100
properties = [
    # "Dissocation energy (nucleofuge)",
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
    "Hardness (eta)",
    # "Hyperhardness (gamma)",
    # "Net Electrophilicity",
    # "Softness (S)"
]
num_labels = len(properties)

# Datasets
data = FingerprintsData(path_data, properties=properties, sanitize=False)

# Split the data into training and validation set
partition = split_dataset(data.fingerprints, data.labels, frac=(0.8, 0.2))

# Model
model = GaussianProcess(partition.features_trainset, partition.labels_trainset.flatten())

# training and validation
researcher = GPModeller(model, data, use_cuda=False, replace_state=True)
researcher.set_optimizer("Adam", lr=0.5)
researcher.set_scheduler(None)
trained_multivariate, expected_train = researcher.train_model(nepoch, partition)

# # Print validation scatterplot
print("validation regression")
multi, label_validset = researcher.validate_model()

create_confidence_plot(
    multi, label_validset.flatten(), properties[0], "validation_scatterplot")

create_scatter_plot(
    multi.mean.reshape(-1, 1), label_validset, properties, "simple_scatterplot")
