#!/usr/bin/env python

import logging
from pathlib import Path

from swan.dataset import GraphData, FingerprintsData
from swan.modeller import Modeller
from swan.modeller.models import FingerprintFullyConnected, MPNN
from swan.utils.log_config import configure_logger
from swan.utils.plot import create_scatter_plot

configure_logger(Path("."))

# Starting logger
LOGGER = logging.getLogger(__name__)


# Path to the DATASET
path_files = Path("tests/files")
path_data = path_files / "cdft_properties.csv"
path_geometries = path_files / "cdft_geometries.json"


#
batch_size = 64
properties = ["Dissocation energy (nucleofuge)"]
nepoch = 50
# Graph NN configuration
# net = MPNN(batch_size=batch_size, output_channels=40)
# data_graph = GraphData(
#     path_data, properties=properties, file_geometries=path_geometries, sanitize=False)

# FullyConnected NN
net = FingerprintFullyConnected()
data_fingerprint = FingerprintsData(
    path_data, properties=properties, sanitize=False)

# training and validation
researcher = Modeller(net, data_fingerprint)
researcher.data.scale_labels()
researcher.train_model(nepoch=nepoch, batch_size=batch_size)
expected, predicted = [researcher.to_numpy_detached(x) for x in researcher.validate_model()]
create_scatter_plot(predicted, expected, properties)
