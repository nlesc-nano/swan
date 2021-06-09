"""class to create models with Pytorch statistical model."""

import logging
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor, nn

from ..dataset.swan_data_base import SwanDataBase
from ..type_hints import PathLike
from ..utils.early_stopping import EarlyStopping
from .base_modeller import BaseModeller

# Starting logger
LOGGER = logging.getLogger(__name__)


class TorchModeller(BaseModeller[torch.Tensor]):
    """Object to create statistical models."""
    def __init__(self,
                 network: nn.Module,
                 data: SwanDataBase,
                 replace_state: bool = False,
                 use_cuda: bool = False):
        """Base class of the modeller

        Parameters
        ----------
        network
            Torch Neural Network [description]
        dataset
            Torch Dataset
        replace_state
            Remove previous state file
        use_cuda
            Train the model using Cuda
        """
        super().__init__(data, replace_state)
        torch.set_default_dtype(torch.float32)
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
        self.set_optimizer('SGD', lr=0.001)

        # set the default loss
        self.set_loss('MSELoss')

        # set scheduler
        self.set_scheduler('StepLR', 0.1)

        # I/O options
        self.workdir = Path('.')
        self.path_scales = self.workdir / "swan_scales.pkl"

        # current number of epoch
        self.epoch = 0

        # Loss data
        self.train_losses = []
        self.validation_losses = []

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
        """Set the loss function for the training.

        Parameters
        ----------
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

    def split_data(self, frac: Tuple[float, float], batch_size: int):
        """Split the data into a training and validation set.

        Parameters
        ----------
        frac
            fraction to divide the dataset, by default [0.8, 0.2]
        """
        # create the dataloader
        indices_train, indices_validate = self.data.create_data_loader(frac=frac, batch_size=batch_size)

        # Store the smiles used for training and validation
        self.state.store_array("smiles_train", self.smiles[indices_train], dtype="str")
        self.state.store_array("smiles_validate", self.smiles[indices_validate], dtype="str")

    def train_model(self,
                    nepoch: int,
                    frac: Tuple[float, float] = (0.8, 0.2),
                    batch_size: int = 64) -> Tuple[Tensor, Tensor]:
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
        self.split_data(frac, batch_size)

        # run over the epochs
        for epoch in range(self.epoch, self.epoch + nepoch):
            LOGGER.info(f"epoch: {epoch}")
            results = []
            expected = []

            # set the model to train mode
            # and init loss
            self.network.train()
            loss_all = 0.

            # iterate over the data loader
            for batch_data in self.data.train_loader:
                x_batch, y_batch = self.data.get_item(batch_data)
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss_batch, predicted = self.train_batch(x_batch, y_batch)
                loss_all += loss_batch
                results.append(predicted)
                expected.append(y_batch)

            # Train loss
            loss = loss_all / len(self.data.train_dataset)
            self.train_losses.append(loss)
            LOGGER.info(f"Loss: {loss}")

            # decrease the LR if necessary
            if self.scheduler is not None:
                self.scheduler.step()

            # Check for early stopping
            self.validate_model()
            self.validation_losses.append(self.validation_loss)
            self.early_stopping(self.save_model, epoch, self.validation_loss)
            if self.early_stopping.early_stop:
                LOGGER.info("EARLY STOPPING")
                break

        # Save the models
        self.save_model(epoch, loss_all)

        # Store the loss
        self.state.store_array("loss_train", self.train_losses)
        self.state.store_array("loss_validate", self.validation_losses)

        return torch.cat(results), torch.cat(expected)

    def train_batch(self, inp_data: Tensor, ground_truth: Tensor) -> Tuple[float, Tensor]:
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

        return loss.item(), prediction

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
                loss_all += loss.item()
                results.append(predicted)
                expected.append(y_val)
            self.validation_loss = loss_all / len(self.data.valid_dataset)
            LOGGER.info(f"validation loss: {self.validation_loss}")
        return torch.cat(results), torch.cat(expected)

    def predict(self, inp_data: Tensor) -> Tensor:
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
        with torch.no_grad():
            self.network.eval()  # Set model to evaluation mode
            predicted = self.network(inp_data)
        return predicted

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

    def load_model(self, filename: PathLike) -> None:
        """Load the model from the state file."""
        checkpoint = torch.load(filename)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
