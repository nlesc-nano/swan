"""Statistical models."""

import logging
import pickle

from abc import abstractmethod

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from flamingo.utils import Options

from sklearn.preprocessing import RobustScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ..utils.early_stopping import EarlyStopping

from ..utils.input_validation import MINIMAL_MODELER_DEFAULTS

# Starting logger
LOGGER = logging.getLogger(__name__)


class ModellerBase:
    """Object to create statistical models."""
    def __init__(self,
                 network: nn.Module,
                 dataset: Dataset,
                 use_cuda: bool = False):
        """Base class of the modeller

        Args:
            network (nn.Module): [description]
            dataset (Dataset): [description]
            opts (Options, optional): [description]. Defaults to None.
        """

        # Set of transformation apply to the dataset
        self.transformer = RobustScaler()

        # Early stopping functionality
        self.early_stopping = EarlyStopping()

        # cuda support
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # create the network
        self.network = network.to(self.device)

        # add dataset in the class
        self.dataset = dataset

        # set the default optimizer
        self.set_optimizer('SGD', lr=0.01)

        # set the default loss
        self.set_loss('MSELoss')

        # I/O options
        self.workdir = './'
        self.path_scales = Path(self.workdir) / "swan_scales.pkl"

        # current number of epoch
        self.epoch = 0

    def set_optimizer(self, name, *args, **kwargs):
        """Set an optimizer using the config file

        Args:
            name ([type]): [description]
        """
        self.optimizer = torch.optim.__getattribute__(name)(
            self.network.parameters(), *args, **kwargs)

    def set_loss(self, name, *args, **kwargs):
        """Set the loss function for the training

        Args:
            name ([type]): [description]
        """
        self.loss_func = getattr(nn, name)(*args, **kwargs)

    @abstractmethod
    def create_data_loader(self, indices: np.ndarray,
                           batch_size: int) -> DataLoader:
        """Create a DataLoader instance for the data."""
        pass

    def train_model(self, nepoch):
        """Train a statistical model."""
        self.network.train()
        LOGGER.info("TRAINING STEP")

        # Set the model to training mode
        self.network.train()

        for epoch in range(self.epoch, self.epoch + nepoch):
            LOGGER.info(f"epoch: {epoch}")
            self.network.train()
            loss_all = 0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss_all += self.train_batch(x_batch, y_batch) * len(x_batch)
            LOGGER.info(f"Loss: {loss_all / self.train_dataset.__len__()}")

            # Check for early stopping
            self.validate_model()
            self.early_stopping(self.save_model, epoch, self.validation_loss)
            if self.early_stopping.early_stop:
                LOGGER.info("EARLY STOPPING")
                break

        # Save the models
        self.save_model(epoch, loss_all)

    def train_batch(self, tensor: Tensor, y_batch: Tensor) -> float:
        """Train a single batch."""
        prediction = self.network(tensor)
        loss = self.loss_func(prediction, y_batch)
        loss.backward()  # backpropagation, compute gradients
        self.optimizer.step()  # apply gradients
        self.optimizer.zero_grad()  # clear gradients for next train

        return loss.item()

    def validate_model(self) -> Tuple[Tensor, Tensor]:
        """Evaluate the model against the validation dataset."""
        results = []
        expected = []

        # Disable any gradient calculation
        with torch.no_grad():
            self.network.eval()
            loss_all = 0
            for x_val, y_val in self.valid_loader:
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)
                predicted = self.network(x_val)
                loss = self.loss_func(predicted, y_val)
                loss_all += loss.item() * len(x_val)
                results.append(predicted)
                expected.append(y_val)
            self.validation_loss = loss_all / self.valid_dataset.__len__()
            LOGGER.info(f"validation loss: {self.validation_loss}")
        return torch.cat(results), torch.cat(expected)

    def predict(self, tensor: Tensor):
        """Use a previously trained model to predict."""
        with torch.no_grad():
            self.network.eval()  # Set model to evaluation mode
            predicted = self.network(tensor)
        return predicted

    def to_numpy_detached(self, tensor: Tensor):
        """Create a view of a Numpy array in CPU."""
        tensor = tensor.cpu() if self.use_cuda else tensor
        return tensor.detach().numpy()

    def scale_labels(self) -> pd.DataFrame:
        """Create a new column with the transformed target."""
        columns = self.dataset.properties
        data = self.dataset.data[columns].to_numpy()
        self.dataset.data[columns] = self.transformer.fit_transform(data)
        self.dump_scale()

    def dump_scale(self) -> None:
        """Save the scaling parameters in a file."""
        with open(self.path_scales, 'wb') as handler:
            pickle.dump(self.transformer, handler)

    def load_scale(self) -> None:
        """Read the scales used for the features."""
        with open(self.path_scales, 'rb') as handler:
            self.transformer = pickle.load(handler)

    def save_model(self,
                   epoch: int,
                   loss: float,
                   filename: str = 'swan_chk.pt') -> None:
        """Save the modle current status."""
        filename = Path(self.workdir) / filename
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss
            }, filename)

    def load_model(self, filename) -> None:
        """Load the model from the state file."""
        checkpoint = torch.load(filename)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
