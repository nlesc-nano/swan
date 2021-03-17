"""Module to process dataset."""
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from flamingo.features.featurizer import generate_fingerprints
from rdkit.Chem import PandasTools
from torch.utils.data import Dataset

from .swan_data_base import SwanDataBase
from .sanitize_data import sanitize_data

PathLike = Union[str, Path]


class FingerprintsData(SwanDataBase):
    def __init__(self,
                 path_data: PathLike,
                 properties: Union[str, List[str]] = None,
                 root: Optional[str] = None,
                 type_fingerprint: str = 'atompair',
                 fingerprint_size: int = 2048,
                 sanitize: bool = False) -> None:
        """generate fingerprint data.

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

        super().__init__()

        self.process_data(path_data,
                          properties=properties,
                          type_fingerprint=type_fingerprint,
                          fingerprint_size=fingerprint_size,
                          sanitize=sanitize)

        self.dataset = FingerprintsDataset(self.fingerprints, self.labels)

        self.data_loader_fun = torch.utils.data.DataLoader

    def process_data(self,
                     path_data: PathLike,
                     properties: Union[str, List[str]] = None,
                     type_fingerprint: str = 'atompair',
                     fingerprint_size: int = 2048,
                     sanitize: bool = False):

        # convert to pd dataFrame if necessaryS
        self.dataframe = pd.read_csv(path_data).reset_index(drop=True)
        PandasTools.AddMoleculeColumnToFrame(self.dataframe,
                                             smilesCol='smiles',
                                             molCol='molecules')

        if sanitize:
            self.dataframe = sanitize_data(self.dataframe)

        self.dataframe.reset_index(drop=True, inplace=True)

        # extract molecules
        self.molecules = self.dataframe['molecules']
        self.properties = properties

        # convert to torch
        if self.properties is not None:

            if not isinstance(self.properties, list):
                self.properties = [self.properties]

            # extract prop to predict
            labels = self.dataframe[self.properties].to_numpy(np.float32)
            size_labels = len(self.molecules)

            self.labels = torch.from_numpy(
                labels.reshape(size_labels, len(self.properties)))
        else:
            self.labels = None

        # compute fingerprints
        fingerprints = generate_fingerprints(self.molecules, type_fingerprint,
                                             fingerprint_size)
        self.fingerprints = torch.from_numpy(fingerprints)

    @staticmethod
    def get_item(batch_data):
        """get the data/ground truth of a minibatch

        Parameters
        ----------
        batch_data : [type]
            data of the mini batch
        """
        return batch_data[0], batch_data[1]


class FingerprintsDataset(Dataset):
    """Read the smiles, properties and compute the fingerprints."""
    def __init__(self, fingerprints, labels) -> None:
        """[summary]

        Parameters
        ----------
        fingerprints : [type]
            [description]
        labels : [type]
            [description]
        """

        self.fingerprints = fingerprints
        self.labels = labels

    def __len__(self) -> int:
        """Return dataset length."""
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return the idx dataset element."""
        return self.fingerprints[idx], self.labels[idx]
