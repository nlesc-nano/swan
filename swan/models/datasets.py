"""Module to process dataset."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..features.featurizer import generate_fingerprints


class FingerprintsDataset(Dataset):
    """Read the smiles, properties and compute the fingerprints."""

    def __init__(
            self, data: pd.DataFrame, property_name: str, type_fingerprint: str,
            fingerprint_size: int) -> tuple:
        """Generate a dataset using fingerprints as features."""
        self.molecules = data['molecules']
        labels = data[property_name].to_numpy(np.float32)
        size_labels = labels.size
        self.labels = torch.from_numpy(labels.reshape(size_labels, 1))
        fingerprints = generate_fingerprints(
            self.molecules, type_fingerprint, fingerprint_size)
        self.fingerprints = torch.from_numpy(fingerprints)

    def __len__(self):
        """Return dataset length."""
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        """Return the idx dataset element."""
        return self.fingerprints[idx], self.labels[idx]
