"""Gaussian Processes modeller."""

import logging
import torch
import gpytorch as gp

from ..dataset.fingerprints_data import FingerprintsData
from ..type_hints import PathLike
from .base_modeller import BaseModeller
from typing import Optional, Tuple


class GPModeller(BaseModeller[torch.Tensor]):
    """Create Gaussian Processes."""

    def __init__(
            self, network: gp.models.GP, data: FingerprintsData, name: str,
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
        super(GPModeller, self).__init__(data, replace_state)
        self.fingerprints = data.fingerprints
        self.labels = data.dataset.labels

        # cuda support
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # create the network
        self.network = network.to(self.device)

        # set the default optimizer
        self.set_optimizer('SGD', lr=0.001)

        # set the default loss
        self.set_loss('MSELoss')


    def train_model(self, nepoch: int, frac: Tuple[float, float] = (0.8, 0.2), **kwargs):
        """Train the model using the given data."""
        # Find optimal model hyperparameters
        self.network.train()
        self.network.likelihood.train()
