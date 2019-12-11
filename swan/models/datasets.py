"""Module to process dataset."""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

from .featurizer import generate_fingerprints, generate_molecular_features


class MolecularDataset(Dataset):
    """Base Dataset class to store molecular properties."""

    def __init__(self, data: pd.DataFrame, property_name: str):
        """Store molecules and labels."""
        self.molecules = data['molecules']
        labels = data[property_name].to_numpy(np.float32)
        size_labels = labels.size
        self.labels = torch.from_numpy(labels.reshape(size_labels, 1))

    def __len__(self):
        """Return dataset length."""
        return self.labels.shape[0]


class MolecularGraphDataset(MolecularDataset):
    """Read Molecular properties represented as graphs."""

    def __init__(self, data: pd.DataFrame, property_name: str):
        """Generate a dataset using molecular graphs as features."""
        super().__init__(data, property_name)

    def __getitem__(self, idx: int):
        """Return the idx dataset element."""
        atomic_features, bond_features = generate_molecular_features(self.molecules[idx])

        # Normalize the features
        atomic_features_l2 = normalize(atomic_features, norm="l2", axis=1)
        bond_features_l2 = normalize(bond_features, norm="l2", axis=1)

        return (atomic_features_l2, bond_features_l2), self.labels[idx]


class FingerprintsDataset(MolecularDataset):
    """Read the smiles, properties and compute the fingerprints."""

    def __init__(
            self, data: pd.DataFrame, property_name: str, type_fingerprint: str,
            fingerprint_size: int) -> tuple:
        """Generate a dataset using fingerprints as features."""
        super().__init__(data, property_name)
        fingerprints = generate_fingerprints(
            self.molecules, type_fingerprint, fingerprint_size)
        self.fingerprints = torch.from_numpy(fingerprints)

    def __getitem__(self, idx: int):
        """Return the idx dataset element."""
        return self.fingerprints[idx], self.labels[idx]
