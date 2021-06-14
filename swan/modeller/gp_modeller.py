"""Gaussian Processes modeller."""

import logging
from typing import Tuple

import gpytorch as gp
import torch
from torch import Tensor

from ..dataset.fingerprints_data import FingerprintsData
from .torch_modeller import TorchModeller
from ..dataset.splitter import SplitDataset

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
            # print("pred: ", self.network.likelihood(prediction).mean[:2], self.labels_trainset[:2])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss = loss.item() / len(self.labels_trainset)
            self.train_losses.append(loss)
            LOGGER.info(f"Loss: {loss}")

            print('Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                loss * len(self.labels_trainset),
                self.network.covar_module.base_kernel.lengthscale.item(),
                self.network.likelihood.noise.item()
            ))
            # # decrease the LR if necessary
            # if self.scheduler is not None:
            #     self.scheduler.step()

            # # Check for early stopping
            # self.validate_model()
            # self.validation_losses.append(self.validation_loss)
            # self.early_stopping(self.save_model, epoch, self.validation_loss)
            # if self.early_stopping.early_stop:
            #     LOGGER.info("EARLY STOPPING")
            #     break

        # Save the models
        self.save_model(epoch, loss)

        # Store the loss
        self.state.store_array("loss_train", self.train_losses)
        # self.state.store_array("loss_validate", self.validation_losses)

        for param_name, param in self.network.named_parameters():
            print(f'Parameter name: {param_name:42} value = {param.item()}')

        return prediction, self.labels_trainset

    def validate_model(self) -> Tuple[Tensor, Tensor]:
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
        return predicted, self.labels_validset
