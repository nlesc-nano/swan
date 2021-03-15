import logging
import tempfile
from typing import Tuple

import numpy as np

import torch
from torch import Tensor
import torch_geometric as tg

from torch.utils.data import DataLoader

from ..dataset import MolGraphDataset
from .modeller_base import ModellerBase

# Starting logger
LOGGER = logging.getLogger(__name__)


class GraphModeller(ModellerBase):
    """Object to create models using molecular graphs."""
    def create_data_loader(self, indices: np.ndarray) -> DataLoader:
        """Create a DataLoader instance for the data."""
        return tg.data.DataLoader(dataset=self.dataset,
                                  batch_size=self.opts.torch_config.batch_size)

    def train_model(self):
        """Train a statistical model."""
        LOGGER.info("TRAINING STEP")
        # Set the model to training mode

        for epoch in range(self.epoch, self.opts.torch_config.epochs):
            LOGGER.info(f"epoch: {epoch}")
            self.network.train()
            loss_all = 0
            for batch in self.train_loader:
                batch.to(self.device)
                loss_batch = self.train_batch(batch, batch.y)
                loss_all += batch.num_graphs * loss_batch
            self.scheduler.step(loss_all / len(self.index_train))

            # Check for early stopping
            self.validate_model()
            self.early_stopping(self.save_model, epoch, self.validation_loss)
            if self.early_stopping.early_stop:
                LOGGER.info("EARLY STOPPING")
                break

            LOGGER.info(f"Training Loss: {loss_all / len(self.index_train)}")

        # Save the models
        self.save_model(epoch, loss_all)

    def validate_model(self) -> Tuple[Tensor, Tensor]:
        """Evaluate the model against the validation dataset."""
        # Disable any gradient calculation
        results = []
        expected = []
        with torch.no_grad():
            self.network.eval()
            loss_all = 0
            for batch in self.valid_loader:
                batch.to(self.device)
                predicted = self.network(batch)
                loss = self.loss_func(predicted, batch.y)
                loss_all += batch.num_graphs * loss.item()
                results.append(predicted)
                expected.append(batch.y)
            self.validation_loss = loss_all / len(self.index_valid)
            LOGGER.info(f"validation loss: {self.validation_loss}")

        return torch.cat(results), torch.cat(expected)
