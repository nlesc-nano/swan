"""Module to process dataset."""
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import torch_geometric as tg
from torch_geometric.data import Data

from .geometry import guess_positions
from .graph.molecular_graph import create_molecular_torch_geometric_graph
from .swan_data_base import SwanDataBase

PathLike = Union[str, Path]


class TorchGeometricGraphData(SwanDataBase):
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
        super().__init__()

        # create the dataframe
        self.dataframe = self.process_data(data_path,
                                           file_geometries=file_geometries)

        # clean the dataframe
        self.clean_dataframe(sanitize=sanitize)

        # Add positions if they don't exists in Dataframe
        if "positions" not in self.dataframe:
            self.dataframe["positions"] = guess_positions(self.dataframe.molecules, optimize_molecule)

        # extract the labels from the dataframe
        self.labels = self.get_labels(properties)
        self.nlabels = self.labels.shape[1]

        # create the graphs
        self.molecular_graphs = self.compute_graph()

        # create the dataset
        self.dataset = GraphDataset(self.molecular_graphs)

        # define the loader type
        self.data_loader_fun = tg.data.DataLoader

    def compute_graph(self) -> List[Data]:
        """compute the graphs in advance."""
        # create the graphs
        molecular_graphs = []
        for idx in range(len(self.labels)):
            gm = create_molecular_torch_geometric_graph(
                self.dataframe["molecules"][idx],
                self.dataframe["positions"][idx],
                labels=self.labels[idx])
            molecular_graphs.append(gm)

        return molecular_graphs

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
        return batch_data, batch_data.y.view(-1, self.nlabels)


class GraphDataset(tg.data.Dataset):
    """Dataset for molecular graphs."""
    def __init__(self, molecular_graphs: List[tg.data.Data]):
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
