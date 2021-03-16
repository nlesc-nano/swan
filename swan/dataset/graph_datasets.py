"""Module to process dataset."""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from rdkit.Chem import PandasTools
from torch_geometric.data import Data

from .geometry import read_geometries_from_files
from .graph.molecular_graph import create_molecular_graph_data
from .sanitize_data import sanitize_data

PathLike = Union[str, Path]


class MolGraphDataset(tg.data.Dataset):
    """Dataset for molecular graphs."""
    def __init__(self,
                 data: PathLike,
                 properties: List[str] = None,
                 root: Optional[str] = None,
                 sanitize: bool = True,
                 file_geometries: Optional[PathLike] = None):
        """Generate a dataset using graphs

        Parameters
        ----------
        data
            path of the csv file
        properties
            Labels names
        root
            Path to the root directory for the dataset
        sanitize
            Check that molecules have a valid conformer
        file_geometries
            Path to a file with the geometries in PDB format

        """
        super().__init__(root)

        # # convert to pd dataFrame if necessary
        self.data = pd.read_csv(data).reset_index(drop=True)

        if file_geometries is not None:
            # i would say that if we want to read the geometry
            # it has to be in the dataframe instead of a separate file
            molecules, positions = read_geometries_from_files(file_geometries)
            self.data["molecules"] = molecules
            self.data["positions"] = positions

        else:

            PandasTools.AddMoleculeColumnToFrame(self.data,
                                                 smilesCol='smiles',
                                                 molCol='molecules')

            if sanitize:
                self.data = sanitize_data(self.data)

            self.data["positions"] = None
            self.data.reset_index(drop=True, inplace=True)

            # extract molecules
            self.molecules = self.data['molecules']
            self.positions = self.data["positions"]

        self.norm = tg.transforms.NormalizeFeatures()

        # get labels
        if properties is not None:
            self.properties = properties
            self.labels = self.data[self.properties].to_numpy(np.float32)
        else:
            self.labels = None

        self.compute_graph()

    def compute_graph(self) -> None:
        """compute the graphs in advance."""
        self.molecular_graphs = []

        for idx in range(self.__len__()):
            labels = None if self.labels is None else torch.Tensor(
                [self.labels[idx]])
            positions = None if self.positions[idx] is None else torch.Tensor(
                self.positions[idx])
            data = create_molecular_graph_data(self.molecules[idx],
                                               positions=positions,
                                               labels=labels)
            self.molecular_graphs.append(data)

    def _download(self):
        pass

    def _process(self):
        pass

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Data:
        """Return the idx dataset element."""
        return self.norm(self.molecular_graphs[idx])
