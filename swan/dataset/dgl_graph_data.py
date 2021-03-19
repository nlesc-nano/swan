"""Interface to build a Dataset for DGL.

see: https://www.dgl.ai/

"""
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from .graph.molecular_graph import create_molecular_dgl_graph
from .swan_data_base import SwanDataBase

# import torch

try:
    import dgl
except ImportError:
    raise ImportError("DGL is a required dependency, see: https://www.dgl.ai/")

# from .geometry import read_geometries_from_files
# from .graph.molecular_graph import create_molecular_graph_data

# import pandas as pd

# import torch_geometric as tg
# from rdkit.Chem import PandasTools
# from torch_geometric.data import Data


PathLike = Union[str, Path]


class DGLGraphData(SwanDataBase):
    """Dataset construction for DGL."""
    def __init__(self,
                 data_path: PathLike,
                 properties: Union[str, List[str]] = None,
                 sanitize: bool = True,
                 file_geometries: Optional[PathLike] = None) -> None:
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
        self.dataset = DGLGraphDataset(self.molecular_graphs, self.labels)

        # define the loader type
        self.data_loader_fun = dgl.dataloading.DataLoader

    def compute_graph(self) -> List[dgl.DGLGraph]:
        """compute the graphs in advance."""

        # initialize positions if they are not in the df
        if "positions" not in self.dataframe:
            self.dataframe["positions"] = None

        # create the graphs
        molecular_graphs = []
        for idx in range(len(self.labels)):
            gm = create_molecular_dgl_graph(
                self.dataframe["molecules"][idx],
                positions=self.dataframe["positions"][idx],
                labels=self.labels[idx])
            molecular_graphs.append(gm)

    def get_item(self, batch_data: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return batch_data[0], batch_data[1]


class DGLGraphDataset(Dataset):
    def __init__(self, molecular_graphs: List[dgl.DGLGraph], labels: torch.Tensor):
        """Generate a dataset using graphs
        """
        super().__init__()
        self.molecular_graphs = molecular_graphs
        self.labels = labels

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.molecular_graphs)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        """Return the idx dataset element."""

        return self.molecular_graphs[idx], self.labels[idx]
