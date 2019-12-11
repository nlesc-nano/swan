"""Module to process dataset."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .featurizer import generate_fingerprints


class MolecularGraphDataset(Dataset):
    """Read Molecular properties represented as graphs."""
    pass


class FingerprintsDataset(Dataset):
    """Read the smiles, properties and compute the fingerprints."""

    def __init__(
            self, df: pd.DataFrame, property_name: str, fingerprint: str,
            fingerprint_size: int) -> tuple:
        molecules = df['molecules']

        fingerprints = generate_fingerprints(
            molecules, fingerprint, fingerprint_size)
        self.fingerprints = torch.from_numpy(fingerprints)
        labels = df[property_name].to_numpy(np.float32)
        size_labels = labels.size
        self.labels = torch.from_numpy(labels.reshape(size_labels, 1))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return self.fingerprints[idx], self.labels[idx]
