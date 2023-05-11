"""Base class representing the data."""
import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import PandasTools
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset, Subset

from swan.type_hints import PathLike
from swan.dataset.geometry import read_geometries_from_files
from swan.dataset.sanitize_data import sanitize_data

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
        data
            filename of the data
        file_geometries
            file containing the geometry of the molecules, by default None

        Returns
        -------
        pd.DataFrame
            data frame
        """
        # create data frame
        dataframe = pd.read_csv(data).reset_index(drop=True)
        dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]

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

    def get_labels(self, properties: Union[str, List[str]]) -> torch.Tensor:
        """extract the labels from the dataframe

        Parameters
        ----------
        properties : List[str]
            names of the properties to extract
        """
        # get labels
        if not isinstance(properties, list):
            properties = [properties]

        if all(p in self.dataframe.columns for p in properties):
            labels = torch.tensor(self.dataframe[properties].to_numpy(
                np.float32)).view(-1, len(properties))
        else:
            msg = f"Not all properties are present in the dataframe. Properties are: {properties}"
            raise RuntimeError(msg)

        return labels

    def clean_dataframe(self, sanitize: bool) -> None:
        """Sanitize the data by removing

        Parameters
        ----------
        sanitize
            Remove molecules without conformer
        """
        if sanitize:
            self.dataframe = sanitize_data(self.dataframe)
        self.dataframe.reset_index(drop=True, inplace=True)

    def create_data_loader(self,
                           frac: Tuple[float, float] = (0.8, 0.2),
                           batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """create the train/valid data loaders using non-overlapping datasets.

        Parameters
        ----------
        frac
            fraction to divide the dataset, by default [0.8, 0.2]
        batch_size
            batchsize, by default 64
        """
        ntotal = len(self.dataset)
        ntrain = int(frac[0] * ntotal)

        indices = np.arange(ntotal)
        np.random.shuffle(indices)

        self.train_dataset = Subset(self.dataset, indices[:ntrain])
        self.valid_dataset = Subset(self.dataset, indices[ntrain:])

        self.train_loader = self.data_loader_fun(dataset=self.train_dataset,
                                                 batch_size=batch_size)

        self.valid_loader = self.data_loader_fun(dataset=self.valid_dataset,
                                                 batch_size=batch_size)

        return indices[:ntrain], indices[ntrain:]

    def scale_labels(self) -> None:
        """Create a new column with the transformed target."""
        self.labels = self.transformer.fit_transform(self.labels)
        self.dump_scale()

    def dump_scale(self) -> None:
        """Save the scaling parameters in a file."""
        with open(self.path_scales, 'wb') as handler:
            pickle.dump(self.transformer, handler)

    def load_scale(self, path_scales: Optional[PathLike] = None) -> None:
        """Read the scales used for the features."""
        path_scales = self.path_scales if path_scales is None else path_scales
        with open(path_scales, 'rb') as handler:
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
