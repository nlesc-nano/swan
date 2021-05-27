
import abc
from typing import Optional, Tuple, TypeVar

from ..dataset.swan_data_base import SwanDataBase
from ..state import StateH5
from ..type_hints import PathLike

T = TypeVar('T')


class BaseModeller:
    """Base class for the modellers."""

    def __init__(self, data: SwanDataBase) -> None:
        self.state = StateH5()
        self.smiles = data.dataframe.smiles.to_numpy()

    @abc.abstractmethod
    def train_model(self, frac: Tuple[float, float] = (0.8, 0.2), **kwargs):
        """Train the model using the given data.

        Parameters
        ----------
        frac
            fraction to divide the dataset, by default [0.8, 0.2]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validate_model(self) -> Tuple[T, T]:
        """compute the output of the model on the validation set

        Returns
        -------
        output of the network, ground truth of the data
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, inp_data: T) -> T:
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

