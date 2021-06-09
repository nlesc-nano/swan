
import abc
from typing import Generic, Optional, Tuple, TypeVar, Union

import numpy as np
import torch

from ..dataset.swan_data_base import SwanDataBase
from ..state import StateH5
from ..type_hints import PathLike

# `bound` preserves all sub-type information, which might be useful
T_co = TypeVar('T_co', bound=Union[np.ndarray, torch.Tensor], covariant=True)


class BaseModeller(Generic[T_co]):
    """Base class for the modellers."""

    def __init__(self, data: SwanDataBase, replace_state: bool) -> None:
        self.state = StateH5(replace_state=replace_state)
        self.smiles = data.dataframe.smiles.to_numpy()

    @abc.abstractmethod
    def train_model(self, nepoch: int, frac: Tuple[float, float] = (0.8, 0.2), **kwargs):
        """Train the model using the given data.

        Parameters
        ----------
        frac
            fraction to divide the dataset, by default [0.8, 0.2]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validate_model(self) -> Tuple[T_co, T_co]:
        """compute the output of the model on the validation set

        Returns
        -------
        output of the network, ground truth of the data
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, inp_data: T_co) -> T_co:
        """compute output of the model for a given input

        Parameters
        ----------
        inp_data
            input data of the network

        Returns
        -------
        Tensor
            output of the network
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, path_model: Optional[PathLike]) -> None:
        """Load the model from the Network file."""
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, *args, **kwargs):
        """Store the trained model."""
        raise NotImplementedError

    def split_fingerprint_data(self, frac: Tuple[float, float]):
        """Split the fingerprint dataset into a training and validation set.

        Parameters
        ----------
        frac
            fraction to divide the dataset, by default [0.8, 0.2]
        """
        # Generate random indices to train and validate the model
        size = len(self.fingerprints)
        indices = np.arange(size)
        np.random.shuffle(indices)

        ntrain = int(size * frac[0])
        self.features_trainset = self.fingerprints[indices[:ntrain]]
        self.features_validset = self.fingerprints[indices[ntrain:]]
        self.labels_trainset = self.labels[indices[:ntrain]]
        self.labels_validset = self.labels[indices[ntrain:]]

        # Store the smiles used for training and validation
        self.state.store_array("smiles_train", self.smiles[indices[:ntrain]], dtype="str")
        self.state.store_array("smiles_validate", self.smiles[indices[ntrain:]], dtype="str")
