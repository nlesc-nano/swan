"""Base class for the Graph data representation."""

from typing import List, Optional, Union


from .geometry import guess_positions
from .swan_data_base import SwanDataBase
from ..type_hints import PathLike


__all__ = ["SwanGraphData"]


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
        if properties is not None:
            self.labels = self.get_labels(properties)
            self.nlabels = self.labels.shape[1]

        # create the graphs
        self.molecular_graphs = self.compute_graph()

    def compute_graph(self):
        """Computhe the graph representing the data."""
        molecular_graphs = []
        for idx in range(len(self.dataframe)):
            labels = None if len(self.labels) == 0 else self.labels[idx]
            gm = self.graph_creator(
                self.dataframe["molecules"][idx],
                self.dataframe["positions"][idx],
                labels=labels)
            molecular_graphs.append(gm)

        return molecular_graphs
