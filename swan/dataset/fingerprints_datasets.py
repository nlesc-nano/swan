"""Module to process dataset."""
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from flamingo.features.featurizer import generate_fingerprints
from rdkit.Chem import PandasTools
from torch.utils.data import Dataset

from .sanitize_data import sanitize_data

PathLike = Union[str, Path]


class FingerprintsDataset(Dataset):
    """Read the smiles, properties and compute the fingerprints."""
    def __init__(self,
                 data: PathLike,
                 properties: Union[str, List[str]] = None,
                 root: Optional[str] = None,
                 type_fingerprint: str = 'atompair',
                 fingerprint_size: int = 2048,
                 sanitize: bool = False) -> None:
        """Generate a dataset using fingerprints as features.

        Parameters
        ----------
        data
            path of the csv file
        properties
            Labels names
        root
            Path to the root directory for the dataset
        type_fingerprint
            Either ``atompair``, ``torsion`` or ``morgan``.
        fingerprint_size
            Size of the fingerprint in bits
        sanitize
            Check that molecules have a valid conformer
        """

        # convert to pd dataFrame if necessaryS
        self.data = pd.read_csv(data).reset_index(drop=True)
        PandasTools.AddMoleculeColumnToFrame(self.data,
                                             smilesCol='smiles',
                                             molCol='molecules')

        if sanitize:
            self.data = sanitize_data(self.data)

        self.data.reset_index(drop=True, inplace=True)

        # extract molecules
        self.molecules = self.data['molecules']

        self.properties = properties
        # convert to torch
        if self.properties is not None:

            if not isinstance(self.properties, list):
                self.properties = [self.properties]

            # extract prop to predict
            labels = self.data[self.properties].to_numpy(np.float32)
            size_labels = len(self.molecules)

            self.labels = torch.from_numpy(
                labels.reshape(size_labels, len(self.properties)))
        else:
            self.labels = None

        # compute fingerprinta
        fingerprints = generate_fingerprints(self.molecules, type_fingerprint,
                                             fingerprint_size)
        self.fingerprints = torch.from_numpy(fingerprints)

    def __len__(self) -> int:
        """Return dataset length."""
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return the idx dataset element."""
        return self.fingerprints[idx], self.labels[idx]
