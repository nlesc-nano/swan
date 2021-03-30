"""Base class for the Graph data representation.

API
---
.. autoclass:: SwanGraphData

"""

from abc import abstractclassmethod
from pathlib import Path
from typing import List, Optional, Union


from .geometry import guess_positions
from .swan_data_base import SwanDataBase


__all__ = ["SwanGraphData"]


PathLike = Union[str, Path]


class SwanGraphData(SwanDataBase):
    """Base class for the Data represented as graphs."""

    def __init__(self,
                 data_path: PathLike,
                 properties: Optional[Union[str, List[str]]] = None,
                 sanitize: bool = True,
                 file_geometries: Optional[PathLike] = None,
                 optimize_molecule: bool = False) -> None:
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
            Perform a molecular optimization using a force field.
        """
        super().__init__()

        # create the dataframe
        self.dataframe = self.process_data(data_path,
                                           file_geometries=file_geometries)

        # clean the dataframe
        self.clean_dataframe(sanitize=sanitize)

        # Add positions if they don't exists in Dataframe
        if "positions" not in self.dataframe:
            self.dataframe["positions"] = guess_positions(
                self.dataframe.molecules, optimize_molecule)

        # extract the labels from the dataframe
        self.labels = self.get_labels(properties)
        self.nlabels = self.labels.shape[1]

        # create the graphs
        self.molecular_graphs = self.compute_graph()

    @abstractclassmethod
    def compute_graph(self):
        """Computhe the graph representing the data."""
        pass
