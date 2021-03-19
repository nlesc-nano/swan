import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import PandasTools
from sklearn.preprocessing import RobustScaler
from torch.utils.data import random_split, Dataset, DataLoader

from .geometry import read_geometries_from_files
from .sanitize_data import sanitize_data

PathLike = Union[str, Path]

__all__ = ["SwanDataBase"]


class SwanDataBase:
    """Base class for the data loaders."""
    def __init__(self) -> None:

        self.dataframe = pd.DataFrame()
        self.dataset = Dataset()  # type: torch.utils.data.Dataset
        self.train_dataset = Dataset()  # type: torch.utils.data.Dataset
        self.valid_dataset = Dataset()  # type: torch.utils.data.Dataset

        self.train_loader = DataLoader(Dataset())  # type: DataLoader
        self.valid_loader = DataLoader(Dataset())  # type: DataLoader
        self.data_loader_fun = DataLoader

        self.labels = torch.tensor([])

        # Set of transformation apply to the dataset
        self.transformer = RobustScaler()

        # I/O options
        self.workdir = Path('.')
        self.path_scales = self.workdir / "swan_scales.pkl"

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

    def get_labels(self, properties: Union[str, List[str],
                                           None]) -> torch.Tensor:
        """extract the labels from the dataframe

        Parameters
        ----------
        properties : List[str]
            names of the properties to extract
        """
        # get labels
        if properties is not None:

            if not isinstance(properties, list):
                properties = [properties]

            labels = torch.tensor(self.dataframe[properties].to_numpy(
                np.float32)).view(-1, len(properties))
        else:
            labels = torch.tensor([None] * self.dataframe.shape[0]).view(-1, 1)

        return labels

    def clean_dataframe(self, sanitize: bool) -> None:
        """Sanitize the data by removing

        Parameters
        ----------
        sanitize : bool, optional
            [description], by default True
        """
        if sanitize:
            self.dataframe = sanitize_data(self.dataframe)
        self.dataframe.reset_index(drop=True, inplace=True)

    def create_data_loader(self,
                           frac=[0.8, 0.2],
                           batch_size: int = 64) -> None:
        """create the train/valid data loaders

        Parameters
        ----------
        frac : list, optional
            fraction to divide the dataset, by default [0.8, 0.2]
        batch_size : int, optional
            batchsize, by default 64
        """
        ntotal = len(self.dataset)
        ntrain = int(frac[0] * ntotal)
        nvalid = ntotal - ntrain

        self.train_dataset, self.valid_dataset = random_split(
            self.dataset, [ntrain, nvalid])

        self.train_loader = self.data_loader_fun(dataset=self.train_dataset,
                                                 batch_size=batch_size)

        self.valid_loader = self.data_loader_fun(dataset=self.valid_dataset,
                                                 batch_size=batch_size)

    def scale_labels(self):
        """Create a new column with the transformed target."""
        self.labels = self.transformer.fit_transform(self.labels)
        self.dump_scale()

    def dump_scale(self) -> None:
        """Save the scaling parameters in a file."""
        with open(self.path_scales, 'wb') as handler:
            pickle.dump(self.transformer, handler)

    def load_scale(self) -> None:
        """Read the scales used for the features."""
        with open(self.path_scales, 'rb') as handler:
            self.transformer = pickle.load(handler)

    def get_item(self, batch_data: Any) -> Tuple[Any, torch.Tensor]:
        """get the data/ground truth of a minibatch

        Parameters
        ----------
        batch_data : [type]
            data of the mini batch

        Raises
        ------
        NotImplementedError
            Not implemented in the base class
        """
        raise NotImplementedError("get item not implemented")
