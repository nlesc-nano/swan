#!/usr/bin/env python
import logging
from pathlib import Path

from swan.dataset import FingerprintsData
from swan.modeller import SKModeller
from swan.utils.log_config import configure_logger
from swan.utils.plot import create_scatter_plot
import sklearn.gaussian_process.kernels as gp

configure_logger(Path("."))

# Starting logger
LOGGER = logging.getLogger(__name__)


# Path to the DATASET
path_data = Path("data/Carboxylic_acids/CDFT/all_carboxylics.csv")

# Scikit learn model hyperparameters
dict_parameters = {
    "tree": {'criterion': 'friedman_mse', 'max_features': 'auto', 'splitter': 'random'},
    "svm": {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'},
    "gaussian": {"kernel": gp.ConstantKernel(1.0, (1e-1, 1e3)) * gp.RBF(10.0, (1e-3, 1e3))}
}


# Training variables
nepoch = 150
batch_size = 32
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
data.scale_labels()

# train and validate
name_model = "tree"
parameters = dict_parameters[name_model]
modeller = SKModeller(data, name_model, **parameters)
modeller.train_model()
predicted, expected = modeller.validate_model()
create_scatter_plot(predicted.reshape(len(expected), 1), expected, properties, f"{name_model}_validation_scatterplot")
