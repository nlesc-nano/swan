"""Module to process dataset."""
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from flamingo.features.featurizer import generate_fingerprints
from torch.utils.data import Dataset

from .graph.molecular_graph import create_molecular_graph_data


class FingerprintsDataset(Dataset):
    """Read the smiles, properties and compute the fingerprints."""

    def __init__(
            self, data: pd.DataFrame, properties: str, type_fingerprint: str,
            fingerprint_size: int) -> None:
        """Generate a dataset using fingerprints as features."""
        self.molecules = data['molecules']
        labels = data[properties].to_numpy(np.float32)
        size_labels = len(self.molecules)
        self.labels = torch.from_numpy(labels.reshape(size_labels, len(properties)))
        fingerprints = generate_fingerprints(
            self.molecules, type_fingerprint, fingerprint_size)
        self.fingerprints = torch.from_numpy(fingerprints)

    def __len__(self) -> int:
        """Return dataset length."""
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return the idx dataset element."""
        return self.fingerprints[idx], self.labels[idx]


class MolGraphDataset(tg.data.Dataset):
    """Dataset for molecular graphs."""

    def __init__(self, root: str, data: pd.DataFrame, properties: List[str] = None):
        """Generate Molecular graph dataset."""
        super().__init__(root)
        data.reset_index(drop=True, inplace=True)
        self.molecules = data['molecules']
        self.positions = data['positions'] if "positions" in data.columns else None

        self.norm = tg.transforms.NormalizeFeatures()
        if properties is not None:
            self.labels = data[properties].to_numpy(np.float32)
        else:
            self.labels = None

    def _download(self):
        pass

    def _process(self):
        pass

    def __len__(self):
        """Return dataset length."""
        return len(self.molecules)

    def __getitem__(self, idx):
        """Return the idx dataset element."""
        labels = None if self.labels is None else torch.Tensor([self.labels[idx]])
        positions = None if self.positions is None else torch.Tensor(self.positions[idx])
        data = create_molecular_graph_data(
            self.molecules[idx], positions=positions, labels=labels)
        return self.norm(data)
