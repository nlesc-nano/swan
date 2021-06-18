"""Gaussian Processes modeller."""

import logging
import warnings
from typing import NamedTuple, Tuple

import gpytorch as gp
import numpy as np
import sklearn
import torch
from torch import Tensor

from ..dataset.fingerprints_data import FingerprintsData
from ..dataset.splitter import SplitDataset
from .torch_modeller import TorchModeller

# Starting logger
LOGGER = logging.getLogger(__name__)


class GPMultivariate(NamedTuple):
    """MultivariateNormal data resulting from the training."""
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


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

        # set the default loss
        self.set_loss()

    def set_loss(self, *args, **kwargs) -> None:
        """Set the loss function for the training."""
        self.loss_func = gp.mlls.ExactMarginalLogLikelihood(self.network.likelihood, self.network)

    def split_data(self, partition: SplitDataset) -> None:
        """Save the smiles used for training and validation."""
        self.features_trainset = partition.features_trainset
        self.features_validset = partition.features_validset

        # Scales the labels coming from the partition
        try:
            self.labels_trainset = torch.from_numpy(self.data.transformer.transform(partition.labels_trainset.numpy()))
            self.labels_validset = torch.from_numpy(self.data.transformer.transform(partition.labels_validset.numpy()))
        except sklearn.exceptions.NotFittedError:
            self.labels_trainset = partition.labels_trainset
            self.labels_validset = partition.labels_validset
            warnings.warn("The labels have not been scaled. Is this the intended behavior?", UserWarning)

        self.store_trainset_in_state(partition.indices, partition.ntrain)

    def train_model(self,
                    nepoch: int,
                    partition: SplitDataset) -> Tuple[Tensor, Tensor]:
        """Train the model

        Parameters
        ----------
        nepoch
            number of ecpoch to run
        partition
            Dataset split into training and validation set
        """

        LOGGER.info("TRAINING STEP")
        self.split_data(partition)

        # run over the epochs
        for epoch in range(self.epoch, self.epoch + nepoch):
            self.network.train()
            self.network.likelihood.train()
            LOGGER.info(f"epoch: {epoch}")

            # set the model to train mode and init loss
            self.network.train()

            output = self.network(self.features_trainset)
            loss = -self.loss_func(output, self.labels_trainset.flatten())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss = loss.item() / len(self.labels_trainset)
            self.train_losses.append(loss)

            lengthscale = self.network.covar_module.base_kernel.lengthscale.item()
            noise = self.network.likelihood.noise.item()
            LOGGER.info(f"Training loss: {loss:.3e}  lengthscale: {lengthscale:.1e} noise: {noise:.1e}")

            # Check for early stopping
            self.validate_model()
            self.validation_losses.append(self.validation_loss)
            self.early_stopping(self.save_model, epoch, self.validation_loss)
            if self.early_stopping.early_stop:
                LOGGER.info("EARLY STOPPING")
                break

            # decrease the LR if necessary
            if self.scheduler is not None:
                self.scheduler.step()

        # Save the models
        self.save_model(epoch, loss)

        # Store the loss
        self.state.store_array("loss_train", self.train_losses)
        self.state.store_array("loss_validate", self.validation_losses)

        predicted = self.network.likelihood(output)
        return self._create_result_object(predicted), self.inverse_transform(self.labels_trainset)

    def validate_model(self) -> Tuple[GPMultivariate, Tensor]:
        """compute the output of the model on the validation set

        Returns
        -------
        Tuple[Tensor, Tensor]
            output of the network, ground truth of the data
        """
        self.network.eval()
        self.network.likelihood.eval()

        # Disable any gradient calculation
        with torch.no_grad(), gp.settings.fast_pred_var():
            output = self.network(self.features_validset)
            loss = -self.loss_func(output, self.labels_validset.flatten())
            self.validation_loss = loss.item() / len(self.labels_validset)
            LOGGER.info(f"validation loss: {self.validation_loss:.3e}")
        return self._create_result_object(self.network.likelihood(output)), self.inverse_transform(self.labels_validset)

    def predict(self, inp_data: Tensor) -> GPMultivariate:
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
        self.network.eval()
        self.network.likelihood.eval()

        with torch.no_grad(), gp.settings.fast_pred_var():
            output = self.network.likelihood(self.network(inp_data))
        return self._create_result_object(output)

    def _create_result_object(self, output: gp.distributions.MultivariateNormal) -> GPMultivariate:
        """Create a NamedTuple with the resulting MultivariateNormal."""
        lower, upper = output.confidence_region()
        return GPMultivariate(*[self.inverse_transform(x).flatten() for x in (output.mean, lower, upper)])
