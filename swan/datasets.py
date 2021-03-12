"""Module to process dataset."""
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from flamingo.features.featurizer import generate_fingerprints
from torch.utils.data import Dataset
from rdkit.Chem import AllChem, PandasTools
from .graph.molecular_graph import create_molecular_graph_data


class FingerprintsDataset(Dataset):
    """Read the smiles, properties and compute the fingerprints."""

    def __init__(
            self, data: Union[pd.DataFrame, str], properties: str, type_fingerprint: str = 'atompair' ,
            fingerprint_size: int = 2048) -> None:
        """Generate a dataset using fingerprints as features.

        Args:
            data (Union): path of the csv file, or pandas data frame containing the data
            properties (str): [description]
            type_fingerprint (str): [description]
            fingerprint_size (int): [description]
        """

        # convert to pd dataFrame if necessary
        if isinstance(data, str):
            data = pd.read_csv(data)
            PandasTools.AddMoleculeColumnToFrame(data, smilesCol='smiles', molCol='molecules')
     
        # extract molecules
        self.molecules = data['molecules']

        # extract prop to predict
        labels = data[properties].to_numpy(np.float32)
        size_labels = len(self.molecules)
        self.labels = torch.from_numpy(labels.reshape(size_labels, len(properties)))
        fingerprints = generate_fingerprints(
            self.molecules, type_fingerprint, fingerprint_size)
        self.fingerprints = torch.from_numpy(fingerprints)

    def create_molecules(self) -> None:
        """Create the molecular representation."""
            


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
