"""Module to process dataset."""
from pathlib import Path
from typing import Any, List, Tuple, Union

import torch
from torch.utils.data import Dataset

from .features.featurizer import generate_fingerprints
from .swan_data_base import SwanDataBase

PathLike = Union[str, Path]


class FingerprintsData(SwanDataBase):
    def __init__(self,
                 path_data: PathLike,
                 properties: Union[str, List[str]] = None,
                 type_fingerprint: str = 'atompair',
                 fingerprint_size: int = 2048,
                 sanitize: bool = True) -> None:
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

        # create the dataframe
        self.dataframe = self.process_data(path_data)

        # clean the dataframe
        self.clean_dataframe(sanitize=sanitize)

        # extract the labels from the dataframe
        self.labels = self.get_labels(properties)
        self.nlabels = self.labels.shape[1]

        # compute fingerprints
        fingerprints = generate_fingerprints(self.dataframe["molecules"],
                                             type_fingerprint,
                                             fingerprint_size)
        self.fingerprints = torch.from_numpy(fingerprints)

        # create the dataset
        self.dataset = FingerprintsDataset(self.fingerprints, self.labels)

        # data loader type
        self.data_loader_fun = torch.utils.data.DataLoader

    def get_item(self, batch_data: List[Any]) -> Tuple[Any, torch.Tensor]:
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
