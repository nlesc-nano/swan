from torch.utils.data import random_split
from sklearn.preprocessing import RobustScaler
from typing import Any, List, Tuple, Union
import torch
import numpy as np
import pickle
from pathlib import Path

from .sanitize_data import sanitize_data


class SwanDataBase:
    """Base class for the data loaders."""
    def __init__(self) -> None:

        self.dataframe = None
        self.dataset = None
        self.train_dataset = None
        self.valid_dataset = None

        self.train_loader = None
        self.valid_loader = None
        self.data_loader_fun = None

        self.labels = None

        # Set of transformation apply to the dataset
        self.transformer = RobustScaler()

        # I/O options
        self.workdir = Path('.')
        self.path_scales = self.workdir / "swan_scales.pkl"

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

    def clean_dataframe(self, sanitize: bool = True) -> None:
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

        ntotal = self.dataset.__len__()
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

    @staticmethod
    def get_item(batch_data: Any) -> Tuple[Any, Any]:
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
