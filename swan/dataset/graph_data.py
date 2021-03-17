"""Module to process dataset."""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch

import torch_geometric as tg
from torch_geometric.data import Data
from rdkit.Chem import PandasTools

from .geometry import read_geometries_from_files
from .graph.molecular_graph import create_molecular_graph_data
from .sanitize_data import sanitize_data
from .swan_data_base import SwanDataBase
PathLike = Union[str, Path]


class GraphData(SwanDataBase):
    """Data loader for graph data."""
    def __init__(self,
                 data_path: PathLike,
                 properties: List[str] = None,
                 sanitize: bool = True,
                 file_geometries: Optional[PathLike] = None):
        """Generate a dataset using graphs

        Parameters
        ----------
        data_path
            path of the csv file
        properties
            Labels names
        sanitize
            Check that molecules have a valid conformer
        file_geometries
            Path to a file with the geometries in PDB format
        """

        super().__init__()

        self.process_data(data_path,
                          properties=properties,
                          sanitize=sanitize,
                          file_geometries=file_geometries)

        self.dataset = GraphDataset(self.molecular_graphs)

        self.data_loader_fun = tg.data.DataLoader

    def process_data(self,
                     data: PathLike,
                     properties: List[str] = None,
                     root: Optional[str] = None,
                     sanitize: bool = True,
                     file_geometries: Optional[PathLike] = None):

        # # convert to pd dataFrame if necessary
        self.dataframe = pd.read_csv(data).reset_index(drop=True)

        if file_geometries is not None:
            # i would say that if we want to read the geometry
            # it has to be in the dataframe instead of a separate file
            molecules, positions = read_geometries_from_files(file_geometries)
            self.dataframe["molecules"] = molecules
            self.dataframe["positions"] = positions

        else:

            PandasTools.AddMoleculeColumnToFrame(self.dataframe,
                                                 smilesCol='smiles',
                                                 molCol='molecules')

            if sanitize:
                self.dataframe = sanitize_data(self.dataframe)

            self.dataframe["positions"] = None
            self.dataframe.reset_index(drop=True, inplace=True)

            # extract molecules
            self.molecules = self.dataframe['molecules']
            self.positions = self.dataframe["positions"]

        # get labels
        if properties is not None:
            self.properties = properties
            self.labels = self.dataframe[self.properties].to_numpy(np.float32)
        else:
            self.labels = None

        self.compute_graph()

    def compute_graph(self) -> None:
        """compute the graphs in advance."""
        self.molecular_graphs = []

        for idx in range(len(self.molecules)):
            labels = None if self.labels is None else torch.Tensor(
                [self.labels[idx]])
            positions = None if self.positions[idx] is None else torch.Tensor(
                self.positions[idx])
            graph = create_molecular_graph_data(self.molecules[idx],
                                                positions=positions,
                                                labels=labels)
            self.molecular_graphs.append(graph)

    @staticmethod
    def get_item(batch_data: List[Data]):
        """get the data/ground truth of a minibatch

        Parameters
        ----------
        batch_data : List[Data]
            data of the mini batch

        Returns
        -------
        [type]
            feature, label
        """
        return batch_data, batch_data.y


class GraphDataset(tg.data.Dataset):
    """Dataset for molecular graphs."""
    def __init__(self, molecular_graphs):
        """Generate a dataset using graphs
        """
        super().__init__()
        self.molecular_graphs = molecular_graphs
        self.normalize_feature = False
        self.norm = tg.transforms.NormalizeFeatures()

    def _download(self):
        pass

    def _process(self):
        pass

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.molecular_graphs)

    def __getitem__(self, idx: int) -> Data:
        """Return the idx dataset element."""
        out = self.molecular_graphs[idx]
        if self.normalize_feature:
            out = self.norm(out)
        return out
