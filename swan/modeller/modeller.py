"""Base class to configure the statistical model."""

import logging
import pickle
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch import Tensor, nn

from ..utils.early_stopping import EarlyStopping
from ..dataset.swan_data import SwanData

# Starting logger
LOGGER = logging.getLogger(__name__)


class Modeller:
    """Object to create statistical models."""
    def __init__(self,
                 network: nn.Module,
                 data: SwanData,
                 use_cuda: bool = False):
        """Base class of the modeller

        Parameters
        ----------
        network
            Torch Neural Network [description]
        dataset
            Torch Dataset
        use_cuda
            Train the model using Cuda
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
        self.data = data

        # set the default optimizer
        self.set_optimizer('SGD', lr=0.01)

        # set the default loss
        self.set_loss('MSELoss')

        # set scheduler
        self.set_scheduler('StepLR', 0.1)

        # I/O options
        self.workdir = Path('.')
        self.path_scales = self.workdir / "swan_scales.pkl"

        # current number of epoch
        self.epoch = 0

    def set_optimizer(self, name: str, *args, **kwargs) -> None:
        """Set an optimizer using the config file

        Parameters
        ----------
        name
            optimizer name

        """
        self.optimizer = torch.optim.__getattribute__(name)(
            self.network.parameters(), *args, **kwargs)

    def set_loss(self, name: str, *args, **kwargs) -> None:
        """Set the loss function for the training

        Parameter
        ---------
        name
            Loss function name

        """
        self.loss_func = getattr(nn, name)(*args, **kwargs)

    def set_scheduler(self, name, *args, **kwargs) -> None:
        """Set the sceduler used for decreasing the LR

        Parameters
        ----------
        name
            Scheduler name

        """
        if name is None:
            self.scheduler = None
        else:
            self.scheduler = getattr(torch.optim.lr_scheduler,
                                     name)(self.optimizer, *args, **kwargs)

    def train_model(self,
                    nepoch: int,
                    frac: List[int] = [0.8, 0.2],
                    batch_size: int = 64) -> None:
        """Train the model

        Parameters
        ----------
        nepoch : int
            number of ecpoch to run
        frac : List[int], optional
            divide the dataset in train/valid, by default [0.8, 0.2]
        batch_size : int, optional
            batchsize, by default 64
        """

        LOGGER.info("TRAINING STEP")

        # create the dataloader
        self.data.create_data_loader(frac=frac, batch_size=batch_size)

        # run over the epochs
        for epoch in range(self.epoch, self.epoch + nepoch):
            LOGGER.info(f"epoch: {epoch}")

            # set the model to train mode
            # and init loss
            self.network.train()
            loss_all = 0.

            # iterate over the data loader
            for batch_data in self.data.train_loader:
                x_batch, y_batch = self.data.get_item(batch_data)
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss_all += self.train_batch(x_batch,
                                             y_batch)  # * len(y_batch)
            LOGGER.info(
                f"Loss: {loss_all / self.data.train_dataset.__len__()}")

            # decrease the LR if necessary
            if self.scheduler is not None:
                self.scheduler.step()

            # Check for early stopping
            self.validate_model()
            self.early_stopping(self.save_model, epoch, self.validation_loss)
            if self.early_stopping.early_stop:
                LOGGER.info("EARLY STOPPING")
                break

        # Save the models
        self.save_model(epoch, loss_all)

    def train_batch(self, inp_data: Tensor, ground_truth: Tensor) -> float:
        """Train a single mini batch

        Parameters
        ----------
        inp_data : Tensor
            input data of the network
        ground_truth : Tensor
            ground trurth of the data points in input

        Returns
        -------
        float
            loss over the mini batch
        """
        prediction = self.network(inp_data)
        loss = self.loss_func(prediction, ground_truth)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def validate_model(self) -> Tuple[Tensor, Tensor]:
        """compute the output of the model on the validation set

        Returns
        -------
        Tuple[Tensor, Tensor]
            output of the network, ground truth of the data
        """
        results = []
        expected = []

        # Disable any gradient calculation
        with torch.no_grad():
            self.network.eval()
            loss_all = 0
            for batch_data in self.data.valid_loader:
                x_val, y_val = self.data.get_item(batch_data)
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

    def predict(self, inp_data: Tensor) -> Tensor:
        """compute output of the model for a given input

        Parameters
        ----------
        inp_data : Tensor
            input data of the network

        Returns
        -------
        Tensor
            output of the network
        """
        with torch.no_grad():
            self.network.eval()  # Set model to evaluation mode
            predicted = self.network(inp_data)
        return predicted

    def to_numpy_detached(self, tensor: Tensor) -> Tensor:
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
        path = self.workdir / filename
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss
            }, path)

    def load_model(self, filename) -> None:
        """Load the model from the state file."""
        checkpoint = torch.load(filename)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']