"""Data representation for torch geometric.

API
---
.. autoclass:: TorchGeometricGraphData

"""
from typing import Any, List, Optional, Tuple, Union

import torch
import torch_geometric as tg
from torch_geometric.data import Data

from swan.type_hints import PathLike
from swan.dataset.data_graph_base import SwanGraphData
from swan.dataset.graph.molecular_graph import create_molecular_torch_geometric_graph


class TorchGeometricGraphData(SwanGraphData):
    """Data loader for graph data."""
    def __init__(self,
                 data_path: PathLike,
                 properties: Optional[Union[str, List[str]]] = None,
                 sanitize: bool = True,
                 file_geometries: Optional[PathLike] = None,
                 optimize_molecule: bool = False
                 ) -> None:
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
        optimize_molecule
            Optimize the geometry if the ``file_geometries`` is not provided

        """
        self.graph_creator = create_molecular_torch_geometric_graph

        super().__init__(
            data_path, properties=properties, sanitize=sanitize,
            file_geometries=file_geometries, optimize_molecule=optimize_molecule)

        # create the dataset
        self.dataset = TorchGeometricGraphDataset(self.molecular_graphs)

        # define the loader type
        self.data_loader_fun = tg.data.DataLoader

    def get_item(self, batch_data: Any) -> Tuple[Any, torch.Tensor]:
        """get the data/ground truth of a minibatch

        Parameters
        ----------
        batch_data
            data of the mini batch

        Returns
        -------
        Tuple with the graph features and the ground true array

        """
        if len(self.labels) == 0:
            return batch_data, None
        else:
            return batch_data, batch_data.y.view(-1, self.nlabels)


class TorchGeometricGraphDataset(tg.data.Dataset):
    """Dataset for molecular graphs."""
    def __init__(self, molecular_graphs: List[tg.data.Data]):
        """Generate a dataset using graphs
        """
        super().__init__()
        self.molecular_graphs = molecular_graphs
        self.normalize_feature = False
        self.norm = tg.transforms.NormalizeFeatures()

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.molecular_graphs)

    def __getitem__(self, idx: int) -> Data:
        """Return the idx dataset element.

        Parameters
        ----------
        idx
            Index of the graph to retrieve

        Returns
        -------
        ``Data`` representing the graph

        """
        # get elements
        out = self.molecular_graphs[idx]

        # normalize if necessary
        if self.normalize_feature:
            out = self.norm(out)

        return out
