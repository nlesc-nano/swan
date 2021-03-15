"""Module to process dataset."""
from typing import Any, Tuple, Union, List

import numpy as np
import pandas as pd
import torch
from flamingo.features.featurizer import generate_fingerprints
from torch.utils.data import Dataset
from rdkit.Chem import PandasTools
from .sanitize_data import sanitize_data


class FingerprintsDataset(Dataset):
    """Read the smiles, properties and compute the fingerprints."""
    def __init__(self,
                 data: str,
                 properties: List[str] = None,
                 type_fingerprint: str = 'atompair',
                 fingerprint_size: int = 2048,
                 sanitize=False) -> None:
        """Generate a dataset using fingerprints as features.

        Args:
            data (Union): path of the csv file, or pandas data frame
                          containing the data
            properties (str): [description]
            type_fingerprint (str): [description]
            fingerprint_size (int): [description]
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
