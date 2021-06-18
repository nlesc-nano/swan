#!/usr/bin/env python

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from swan.dataset import FingerprintsData, load_split_dataset
from swan.modeller import GPModeller
from swan.modeller.models import GaussianProcess
from swan.utils.log_config import configure_logger

# Starting logger
configure_logger(Path("."))
LOGGER = logging.getLogger(__name__)

# Set float size default
torch.set_default_dtype(torch.float32)

partition = load_split_dataset()
features, labels = [torch.from_numpy(getattr(partition, x).astype(np.float32)) for x in ("features_trainset", "labels_trainset")]
model = GaussianProcess(features, labels.flatten())
data = FingerprintsData(Path("tests/files/smiles.csv"), properties=None, sanitize=False)

researcher = GPModeller(model, data, use_cuda=False, replace_state=False)
# # If the labels are scaled you need to load the scaling functionality
# researcher.data.load_scale()
researcher.load_model("swan_chk.pt")

fingers = data.fingerprints
predicted = researcher.predict(fingers)
df = pd.DataFrame(
    {"smiles": data.dataframe.smiles, "mean": predicted.mean, "lower": predicted.lower, "upper": predicted.upper})

df.to_csv("predicted_values.csv")
