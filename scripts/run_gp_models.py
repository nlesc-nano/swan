#!/usr/bin/env python

import logging
from pathlib import Path
import torch
from swan.dataset import FingerprintsData, split_dataset
from swan.modeller import GPModeller
from swan.modeller.models import GaussianProcess
from swan.utils.log_config import configure_logger
from swan.utils.plot import create_scatter_plot


configure_logger(Path("."))

# Starting logger
LOGGER = logging.getLogger(__name__)


# Path to the DATASET
path_files = Path("tests/files")
path_data = path_files / "thousand.csv"

# Training variables
nepoch = 100
batch_size = 32
properties = ["gammas"]
# properties = [
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
    # "Hardness (eta)",
    # "Hyperhardness (gamma)",
    # "Net Electrophilicity",
    # "Softness (S)"
# ]
num_labels = len(properties)

# Datasets
data = FingerprintsData(
    path_data, properties=properties, sanitize=False)

# Split the data into training and validation set
partition = split_dataset(data.fingerprints, data.labels, frac=(0.8, 0.2))

# Model
model = GaussianProcess(partition.features_trainset, partition.labels_trainset.flatten())


# training and validation
torch.set_default_dtype(torch.float32)
researcher = GPModeller(model, data, use_cuda=False)
researcher.set_optimizer("Adam", lr=0.1)
researcher.set_scheduler("StepLR", 0.1)
researcher.data.scale_labels()
multivariate_normal, expected = researcher.train_model(nepoch, partition, batch_size=batch_size)
# predicted_train, expected_train = [x.cpu().detach().numpy() for x in trained_data]
# print("train regression")
# create_scatter_plot(predicted_train, expected_train, properties, "trained_scatterplot")

# # Print validation scatterplot
# print("validation regression")
# predicted_validation, expected_validation = [x.cpu().detach().numpy() for x in researcher.validate_model()]
# create_scatter_plot(predicted_validation, expected_validation, properties, "validation_scatterplot")

# print("properties stored in the HDF5")
# researcher.state.show()
