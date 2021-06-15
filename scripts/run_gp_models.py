#!/usr/bin/env python

import logging
from pathlib import Path
import torch
from swan.dataset import FingerprintsData, split_dataset
from swan.modeller import GPModeller
from swan.modeller.models import GaussianProcess
from swan.utils.log_config import configure_logger
from swan.utils.plot import create_confidence_plot

# Starting logger
configure_logger(Path("."))
LOGGER = logging.getLogger(__name__)

# Set float size default
torch.set_default_dtype(torch.float32)

# Path to the DATASET
path_data = Path("cdft.csv")

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
data = FingerprintsData(
    path_data, properties=properties, sanitize=False)

# Split the data into training and validation set
partition = split_dataset(data.fingerprints, data.labels, frac=(0.8, 0.2))

# Model
model = GaussianProcess(partition.features_trainset, partition.labels_trainset.flatten())

# training and validation
researcher = GPModeller(model, data, use_cuda=False, replace_state=True)
researcher.set_optimizer("Adam", lr=0.1)
researcher.set_scheduler("StepLR", 0.1)
researcher.data.scale_labels()
trained_multivariate, expected_train = researcher.train_model(nepoch, partition)

# # Print validation scatterplot
print("validation regression")
multi, label_validset = researcher.validate_model()

create_confidence_plot(
    multi, label_validset.flatten(), properties[0], "validation_scatterplot")

print("properties stored in the HDF5")
researcher.state.show()


fingers = FingerprintsData(Path("tests/files/smiles.csv"), properties=None, sanitize=False)

predicted = researcher.predict(fingers.fingerprints)
print("mean: ", predicted.mean)
print("lower: ", predicted.lower)
print("upper: ", predicted.upper)
