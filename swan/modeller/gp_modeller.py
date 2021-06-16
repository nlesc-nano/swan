"""Gaussian Processes modeller."""

import logging
from typing import NamedTuple, Tuple

import gpytorch as gp
import torch
from torch import Tensor

from ..dataset.fingerprints_data import FingerprintsData
from ..dataset.splitter import SplitDataset
import numpy as np
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
        self.labels_trainset = torch.from_numpy(self.data.transformer.transform(partition.labels_trainset.numpy()))
        self.labels_validset = torch.from_numpy(self.data.transformer.transform(partition.labels_validset.numpy()))

        indices = partition.indices
        ntrain = partition.ntrain
        self.state.store_array("smiles_train", self.smiles[indices[:ntrain]], dtype="str")
        self.state.store_array("smiles_validate", self.smiles[indices[ntrain:]], dtype="str")

    def train_model(self,
                    nepoch: int,
                    partition: SplitDataset,
                    batch_size: int = 64) -> Tuple[Tensor, Tensor]:
        """Train the model

        Parameters
        ----------
        nepoch
            number of ecpoch to run
        partition
            Dataset split into training and validation set
        batch_size
            batchsize, by default 64
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

            prediction = self.network(self.features_trainset)
            loss = -self.loss_func(prediction, self.labels_trainset.flatten())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss = loss.item() / len(self.labels_trainset)
            self.train_losses.append(loss)
            LOGGER.info(f"Loss: {loss}")

            LOGGER.info('Loss: %.1e   lengthscale: %.1e   noise: %.1e' % (
                loss * len(self.labels_trainset),
                self.network.covar_module.base_kernel.lengthscale.item(),
                self.network.likelihood.noise.item()
            ))
            # decrease the LR if necessary
            if self.scheduler is not None:
                self.scheduler.step()

        # Save the models
        self.save_model(epoch, loss)

        # Store the loss
        self.state.store_array("loss_train", self.train_losses)

        return self._create_result_object(prediction), self.inverse_transform(self.labels_trainset)

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
            predicted = self.network.likelihood(self.network(self.features_validset))
            loss = -self.loss_func(predicted, self.labels_validset.flatten())
            self.validation_loss = loss.item() / len(self.features_validset)
            LOGGER.info(f"validation loss: {self.validation_loss}")
        return self._create_result_object(predicted), self.inverse_transform(self.labels_validset)

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
            predicted = self.network.likelihood(self.network(inp_data))
        return self._create_result_object(predicted)

    def _create_result_object(self, output: gp.distributions.MultivariateNormal) -> GPMultivariate:
        """Create a NamedTuple with the resulting MultivariateNormal."""
        lower, upper = output.confidence_region()
        return GPMultivariate(*[self.inverse_transform(x).flatten() for x in (output.mean, lower, upper)])
