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
path_data = Path("cdft.csv")

# Training variables
nepoch = 50
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
torch.set_default_dtype(torch.float32)
researcher = GPModeller(model, data, use_cuda=False, replace_state=True)
researcher.set_optimizer("Adam", lr=0.5)
# researcher.set_scheduler("StepLR", 0.1)
researcher.data.scale_labels()
trained_data = researcher.train_model(nepoch, partition)
# predicted_train, expected_train = [x.detach().numpy() for x in trained_data]
# print("train regression")
# create_scatter_plot(predicted_train, expected_train, properties, "trained_scatterplot")

# # Print validation scatterplot
print("validation regression")
output, label_validset = researcher.validate_model()
lower, upper = output.confidence_region()
print("Lower: ", lower[:5].detach())
print("Upper: ", upper[:5].detach())
print("means")
print(output.mean[:10])
print("ground true")
print(label_validset[:10])
create_scatter_plot(output.mean.unsqueeze(-1).numpy(), label_validset.numpy(), properties, "validation_scatterplot")

# print("properties stored in the HDF5")
# researcher.state.show()
