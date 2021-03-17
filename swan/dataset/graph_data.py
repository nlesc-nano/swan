"""Module to process dataset."""
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

import torch_geometric as tg
from torch_geometric.data import Data
from rdkit.Chem import PandasTools

from .geometry import read_geometries_from_files
from .graph.molecular_graph import create_molecular_graph_data

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

        # create the dataframe
        self.dataframe = self.process_data(data_path,
                                           file_geometries=file_geometries)

        # clean the dataframe
        self.clean_dataframe(sanitize=sanitize)

        # extract the labels from the dataframe
        self.labels = self.get_labels(properties)
        self.nlabels = self.labels.shape[1]

        # create the graphs
        self.molecular_graphs = self.compute_graph()

        # create the dataset
        self.dataset = GraphDataset(self.molecular_graphs)

        # define the loader type
        self.data_loader_fun = tg.data.DataLoader

    def process_data(
            self,
            data: PathLike,
            file_geometries: Optional[PathLike] = None) -> pd.DataFrame:
        """process the data frame

        Parameters
        ----------
        data : PathLike
            filename of the data
        file_geometries : Optional[PathLike], optional
            file containing the geometry of the molecules, by default None

        Returns
        -------
        pd.DataFrame
            data frame
        """

        # create data frame
        dataframe = pd.read_csv(data).reset_index(drop=True)

        # read geometries from file
        if file_geometries is not None:
            # i would say that if we want to read the geometry
            # it has to be in the dataframe instead of a separate file
            molecules, positions = read_geometries_from_files(file_geometries)
            dataframe["molecules"] = molecules
            dataframe["positions"] = positions

        # ignore geometries
        # do not initialize positions as sanitize_data
        # will then erase all entries
        else:
            PandasTools.AddMoleculeColumnToFrame(dataframe,
                                                 smilesCol='smiles',
                                                 molCol='molecules')

        return dataframe

    def compute_graph(self) -> None:
        """compute the graphs in advance."""

        # initialize positions if they are not in the df
        if "positions" not in self.dataframe:
            self.dataframe["positions"] = None

        # create the graphs
        molecular_graphs = []
        for idx in range(len(self.labels)):
            gm = create_molecular_graph_data(
                self.dataframe["molecules"][idx],
                positions=self.dataframe["positions"][idx],
                labels=self.labels[idx])
            molecular_graphs.append(gm)

        return molecular_graphs

    def get_item(self, batch_data: List[Data]):
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
        return batch_data, batch_data.y.view(-1, self.nlabels)


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

        # get elements
        out = self.molecular_graphs[idx]

        # normalize if necessary
        if self.normalize_feature:
            out = self.norm(out)

        return out
