"""Statistical models."""
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as fun
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from swan.log_config import config_logger

from .featurizer import generate_fingerprints
from .input_validation import validate_input
from .plot import create_scatter_plot

__all__ = ["Modeler"]

# Starting logger
LOGGER = logging.getLogger(__name__)


def main():
    """Parse the command line arguments and call the modeler class."""
    parser = argparse.ArgumentParser(description="modeler -i input.yml")
    # configure logger
    parser.add_argument('-i', required=True,
                        help="Input file with options")
    parser.add_argument("-m", "--mode", help="Operation mode: train or predict",
                        choices=["train", "predict"], default="train")
    parser.add_argument('-w', help="workdir", default=".")
    args = parser.parse_args()

    # start logger
    config_logger(Path(args.w))

    # log date
    LOGGER.info(f"Starting at: {datetime.now()}")

    # Check that the input is correct
    opts = validate_input(Path(args.i))
    opts.mode = args.mode

    if args.mode == "train":
        train_and_validate_model(opts)
    else:
        predict_properties(opts)


class Net(torch.nn.Module):
    """Create a Neural network object using Pytorch."""

    def __init__(self, n_feature: int, n_hidden: int, n_output: int):
        super(Net, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(n_feature, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_output)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """Activation function for hidden layer."""
        return self.seq(tensor)


class LigandsDataset(Dataset):
    """Read the smiles, properties and compute the fingerprints."""

    def __init__(self, df: pd.DataFrame, property_name: str) -> tuple:
        smiles = df['smiles'].to_numpy()
        fingerprints = generate_fingerprints(smiles)
        self.fingerprints = torch.from_numpy(fingerprints)
        labels = df[property_name].to_numpy(np.float32)
        size_labels = labels.size
        self.labels = torch.from_numpy(labels.reshape(size_labels, 1))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return self.fingerprints[idx], self.labels[idx]


class Modeler:
    """Object to create statistical models."""

    def __init__(self, opts: dict, load_model: bool = False):
        """Set up a modeler object."""
        self.opts = opts
        self.data = pd.read_csv(opts.dataset_file, index_col=0).reset_index()
        self.sanitize_data()

        if opts.use_cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        if not load_model:
            self.create_new_model()
        else:
            self.load_trained_model()

    def sanitize_data(self):
        """Check that the data in the DataFrame is valid."""
        self.data.dropna(inplace=True)

    def create_new_model(self):
        """Configure a new model."""
        # Create an Network architecture
        self.network = Net(n_feature=2048, n_hidden=1024, n_output=1)
        self.network = self.network.to(self.device)

        # Create an optimizer
        self.optimizer = torch.optim.SGD(
            self.network.parameters(), **self.opts.torch_config.optimizer)

        # Create loss function
        self.loss_func = torch.nn.MSELoss()

    def load_data(self):
        """Create loaders for the train and validation dataset."""
        self.train_loader = self.create_data_loader(self.index_train)
        self.valid_loader = self.create_data_loader(self.index_valid)

    def create_data_loader(self, indices: np.array) -> DataLoader:
        """Create a DataLoader instance for the data."""
        dataset = LigandsDataset(self.data.loc[indices], 'normalized_labels')
        return DataLoader(
            dataset=dataset, batch_size=self.opts.torch_config.batch_size)

    def train_model(self):
        """Train an statistical model."""
        LOGGER.info("TRAINING STEP")

        # Set the model to training mode
        self.network.train()

        for epoch in range(self.opts.torch_config.epochs):
            loss_batch = 0
            for x_batch, y_batch in self.train_loader:
                if self.opts.use_cuda:
                    x_batch = x_batch.to('cuda')
                    y_batch = y_batch.to('cuda')
                loss_batch = self.train_batch(x_batch, y_batch)
            mean = loss_batch / self.opts.torch_config.batch_size
            if epoch % self.opts.torch_config.frequency_log_epochs == 0:
                LOGGER.info(f"Loss: {mean}")

        # Save the models
        torch.save(self.network.state_dict(), self.opts.model_path)

    def train_batch(self, x_batch: Variable, y_batch: Variable) -> float:
        """Train a single batch."""
        prediction = self.network(x_batch)
        loss = self.loss_func(prediction, y_batch)
        loss.backward()              # backpropagation, compute gradients
        self.optimizer.step()        # apply gradients
        self.optimizer.zero_grad()   # clear gradients for next train

        cpu_tensor = loss.data.cpu()

        return cpu_tensor.numpy()

    def evaluate_model(self):
        """Evaluate the model against the validation dataset."""
        LOGGER.info("VALIDATION STEP")
        # Disable any gradient calculation
        with torch.no_grad():
            self.network.eval()
            val_loss = 0
            for x_val, y_val in self.valid_loader:
                if self.opts.use_cuda:
                    x_val = x_val.to('cuda')
                    y_val = y_val.to('cuda')
                predicted = self.network(x_val)
                val_loss += self.loss_func(y_val, predicted)
            mean_val_loss = val_loss / self.opts.torch_config.batch_size
        LOGGER.info(f"validation loss:{mean_val_loss}")

    def predict(self, tensor: Tensor):
        """Use a previously trained model to predict."""
        with torch.no_grad():
            self.network.load_state_dict(torch.load(self.opts.model_path))
            self.network.eval()  # Set model to evaluation mode
            predicted = self.network(tensor)
        return predicted

    def plot_evaluation(self):
        """Create a scatter plot of the predict values vs the ground true."""
        dataset = self.valid_loader.dataset
        tensor_features = dataset.fingerprints
        if self.opts.use_cuda:
            tensor_features = tensor_features.to('cuda')
        result = self.network(tensor_features)
        result = result.cpu() if self.opts.use_cuda else result
        predicted = result.detach().numpy()
        expected = np.stack(dataset.labels).flatten()
        create_scatter_plot(predicted, expected)

    def split_data(self, frac: float = 0.2):
        """Split the data into a training and test set."""
        size_valid = int(self.data.index.size * frac)
        self.index_valid = np.random.choice(self.data.index, size=size_valid)
        self.index_train = np.setdiff1d(self.data.index, self.index_valid, assume_unique=True)

    def normalize_data(self) -> pd.DataFrame:
        """Create a new column with the normalized target."""
        self.data['normalized_labels'] = self.data[self.opts.property] / \
            np.linalg.norm(self.data[self.opts.property])


def train_and_validate_model(opts: dict) -> None:
    """Train the model usign the data specificied by the user."""
    researcher = Modeler(opts)
    researcher.normalize_data()
    researcher.split_data()
    researcher.load_data()
    researcher.train_model()
    researcher.evaluate_model()
    researcher.plot_evaluation()


def predict_properties(opts: dict) -> Tensor:
    """Use a previous trained model to predict properties."""
    pass
    # return researcher.predict(x)
