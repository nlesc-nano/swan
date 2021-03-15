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
                 opts: Options = None):
        """Set up a modeler object."""

        if opts is None:
            self.opts = Options(MINIMAL_MODELER_DEFAULTS)
        else:
            self.opts = Options(opts)

        # Set of transformation apply to the dataset
        self.transformer = RobustScaler()

        # Early stopping functionality
        self.early_stopping = EarlyStopping()

        if self.opts.use_cuda and self.opts.mode == "train":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.network = network.to(self.device)
        self.dataset = dataset
        self.configure()

    def configure(self):
        """Configure a new model."""
        self.epoch = 0
        self.set_optimizer()

        # Scales for the features
        self.path_scales = Path(self.opts.workdir) / "swan_scales.pkl"

        # Reload model from file
        if self.opts.restart or self.opts.mode == "predict":
            self.load_model()

        # Create loss function
        self.loss_func = getattr(nn, self.opts.torch_config.loss_function)()

    def set_optimizer(self) -> None:
        """Select the optimizer."""
        optimizers = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
        config = self.opts.torch_config.optimizer
        fun = optimizers[config["name"]]
        if config["name"] == "sgd":
            self.optimizer = fun(self.network.parameters(),
                                 lr=config["lr"],
                                 momentum=config["momentum"],
                                 nesterov=config["nesterov"])
        else:
            self.optimizer = fun(self.network.parameters(), lr=config["lr"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', min_lr=0.00001)

    def load_data(self):
        """Create loaders for the train and validation dataset."""
        self.train_loader = self.create_data_loader(self.index_train)
        self.valid_loader = self.create_data_loader(self.index_valid)

    @abstractmethod
    def create_data_loader(self, indices: np.ndarray) -> DataLoader:
        """Create a DataLoader instance for the data."""
        pass

    def train_model(self):
        """Train a statistical model."""
        self.network.train()
        LOGGER.info("TRAINING STEP")

        # Set the model to training mode
        self.network.train()

        for epoch in range(self.epoch, self.opts.torch_config.epochs):
            LOGGER.info(f"epoch: {epoch}")
            self.network.train()
            loss_all = 0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss_all += self.train_batch(x_batch, y_batch) * len(x_batch)
            LOGGER.info(f"Loss: {loss_all / len(self.index_train)}")

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
            self.validation_loss = loss_all / len(self.index_valid)
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
        tensor = tensor.cpu() if self.opts.use_cuda else tensor
        return tensor.detach().numpy()

    def split_data(self, frac: float = 0.2):
        """Split the data into a training and test set."""
        size_valid = int(len(self.data.index) * frac)
        # Sample the indices without replacing
        self.index_valid = np.random.choice(self.data.index,
                                            size=size_valid,
                                            replace=False)
        self.index_train = np.setdiff1d(self.data.index,
                                        self.index_valid,
                                        assume_unique=True)

    def scale_labels(self) -> pd.DataFrame:
        """Create a new column with the transformed target."""
        columns = self.opts.properties
        if self.opts.scale_labels:
            data = self.data[columns].to_numpy()
            self.data[columns] = self.transformer.fit_transform(data)
            self.dump_scale()

    def dump_scale(self) -> None:
        """Save the scaling parameters in a file."""
        with open(self.path_scales, 'wb') as handler:
            pickle.dump(self.transformer, handler)

    def load_scale(self) -> None:
        """Read the scales used for the features."""
        with open(self.path_scales, 'rb') as handler:
            self.transformer = pickle.load(handler)

    def save_model(self, epoch: int, loss: float) -> None:
        """Save the modle current status."""
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss
            }, self.opts.model_path)

    def load_model(self) -> None:
        """Load the model from the state file."""
        checkpoint = torch.load(self.opts.model_path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
