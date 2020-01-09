"""Module to process dataset."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch_geometric as tg
from ..features.featurizer import generate_fingerprints
from ..graph.molecular_graph import create_molecular_graph_data


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


class MolGraphDataset(tg.data.Dataset):
    """Dataset for molecular graphs."""

    def __init__(self, root: str, data: pd.DataFrame, property_name: str = None):
        """Generate Molecular graph dataset."""
        super().__init__(root)
        self.molecules = data['molecules']
        self.molecules.reset_index(drop=True, inplace=True)
        self.norm = tg.transforms.NormalizeFeatures()
        if property_name is not None:
            self.labels = data[property_name].to_numpy(np.float32)
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
        if self.labels is not None:
            labels = torch.Tensor([self.labels[idx]]).reshape(1, 1)
        else:
            labels = None
        data = create_molecular_graph_data(self.molecules[idx], labels)
        return self.norm(data)
