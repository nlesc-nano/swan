"""Gaussian Processes modeller."""

import logging
from typing import Optional, Tuple

import gpytorch as gp
import torch
from torch import Tensor

from ..dataset.fingerprints_data import FingerprintsData
from ..type_hints import PathLike
from .base_modeller import BaseModeller
from .torch_modeller import TorchModeller

# Starting logger
LOGGER = logging.getLogger(__name__)


class GPModeller(TorchModeller):
    """Create Gaussian Processes."""

    def __init__(
            self, network: gp.models.GP, data: FingerprintsData,
            replace_state: bool = False, use_cuda: bool = False):
        """Create GPModeller instance

        Parameters
        ----------
        network
            GPytorch Model
        dataset
            Torch Dataset
        replace_state
            Remove previous state file
        use_cuda
            Train the model using Cuda
        """
        super(GPModeller, self).__init__(
            network, data, replace_state=replace_state, use_cuda=use_cuda)

        # Store the attributes and labels to create the training and validation sets
        self.fingerprints = data.fingerprints
        self.labels = data.dataset.labels

        # set the default loss
        self.set_loss()

    def set_loss(self, *args, **kwargs) -> None:
        """Set the loss function for the training."""
        model = self.network
        likelihood = model.likelihood
        self.loss_func = gp.mlls.ExactMarginalLogLikelihood(likelihood, model)

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

        # Split data in training and validation set
        self.split_fingerprint_data(frac)

        # run over the epochs
        for epoch in range(self.epoch, self.epoch + nepoch):
            LOGGER.info(f"epoch: {epoch}")

            # set the model to train mode and init loss
            self.network.train()

            prediction = self.network(self.features_trainset)
            loss = self.loss_func(prediction, self.labels_trainset)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss = loss.item() / len(self.data.train_dataset)
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
        self.save_model(epoch, loss)

        # Store the loss
        self.state.store_array("loss_train", self.train_losses)
        self.state.store_array("loss_validate", self.validation_losses)

        return prediction, self.labels_trainset

    def validate_model(self) -> Tuple[Tensor, Tensor]:
        """compute the output of the model on the validation set

        Returns
        -------
        Tuple[Tensor, Tensor]
            output of the network, ground truth of the data
        """
        # Disable any gradient calculation
        with torch.no_grad():
            self.network.eval()
            predicted = self.network(self.features_validset)
            loss = self.loss_func(predicted, self.labels_validset)
            self.validation_loss = loss.item() / len(self.data.valid_dataset)
            LOGGER.info(f"validation loss: {self.validation_loss}")
        return predicted, self.labels_validset
