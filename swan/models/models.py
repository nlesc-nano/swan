"""Statistical models."""
from datetime import datetime
from pathlib import Path
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as fun

from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import (DataLoader, TensorDataset, random_split)
from swan.log_config import config_logger
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
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x: Tensor) -> Tensor:
        """Activation function for hidden layer."""
        x = fun.relu(self.hidden(x))
        # linear output
        x = self.predict(x)
        return x


class Modeler:
    """Object to create statistical models."""

    def __init__(self, opts: dict):
        self.opts = opts

        # Create an Network architecture
        self.network = Net(n_feature=1, n_hidden=10, n_output=1)

        # Create an optimizer
        self.optimizer = torch.optim.SGD(
            self.network.parameters(), **self.opts.torch_config.optimizer)

        # Create loss function
        self.loss_func = torch.nn.MSELoss()

    def train_model(self, train_loader: DataLoader) -> None:
        """Train an statistical model."""
        LOGGER.info("TRAINING STEP")

        # Set the model to training mode
        self.network.train()

        for t in range(self.opts.torch_config.epochs):
            loss_batch = 0
            for x_batch, y_batch in train_loader:
                loss_batch = self.train_batch(x_batch, y_batch)
            mean = loss_batch / self.opts.torch_config.batch_size
            if t % self.opts.torch_config.frequency_log_epochs == 0:
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

        return loss.data.numpy()

    def evaluate_model(self, valid_loader: DataLoader) -> None:
        """Evaluate the model against the validation dataset."""
        LOGGER.info("VALIDATION STEP")
        # Disable any gradient calculation
        with torch.no_grad():
            self.network.eval()
            val_loss = 0
            for x_val, y_val in valid_loader:
                predicted = self.network(x_val)
                val_loss += self.loss_func(y_val, predicted)
            mean_val_loss = val_loss / self.opts.torch_config.batch_size
        LOGGER.info(f"validation loss:{mean_val_loss}")

    def predict(self, x: Tensor):
        """Use a previously trained model to predict."""
        with torch.no_grad():
            self.network.load_state_dict(torch.load(self.opts.model_path))
            self.network.eval()  # Set model to evaluation mode
            predicted = self.network(x)
        return predicted

    def plot_evaluation(self, train_loader, valid_loader) -> None:
        """Create a scatter plot of the predict values vs the ground true."""
        dataset = valid_loader.dataset
        index = dataset.indices
        predicted = self.network(dataset.dataset[index][0]).detach().numpy()
        expected = np.stack(dataset.dataset[index][1]).flatten()
        create_scatter_plot(predicted, expected)


def train_and_validate_model(opts: dict) -> None:
    """Train the model usign the data specificied by the user."""
    researcher = Modeler(opts)
    train_loader, valid_loader = load_data(opts)
    researcher.train_model(train_loader)
    researcher.evaluate_model(valid_loader)
    researcher.plot_evaluation(train_loader, valid_loader)


def load_data(opts: dict) -> tuple:
    """Load the data and split it into a training and validation set."""
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    dataset = TensorDataset(x, y)
    train_data, valid_data = random_split(dataset, [80, 20])

    # Generate the classes that feed the data for training and validation
    train_loader = DataLoader(
        dataset=train_data, batch_size=opts.torch_config.batch_size, shuffle=True)

    valid_loader = DataLoader(
        dataset=valid_data, batch_size=opts.torch_config.batch_size)

    return train_loader, valid_loader


def predict_properties(opts: dict) -> Tensor:
    """Use a previous trained model to predict properties."""
    x = torch.unsqueeze(torch.linspace(-1, 1, 10), dim=1)
    researcher = Modeler(opts)
    return researcher.predict(x)


#     def select_featurizer(self) -> None:
#         """
#         Use featurizer chosen by the user
#         """
#         logger.info(f"Using featurizer:{self.opts.featurizer}")
#         names = {
#             "circularfingerprint": "CircularFingerprint"
#         }
#         feat = import_module("deepchem.feat")
#         featurizer = getattr(feat, names[self.opts.featurizer])
#         self.featurizer = featurizer()

#     def select_metric(self) -> None:
#         """
#         Create instances of the metric to use
#         """
#         # Import the metric
#         mod_metric = import_module("deepchem.metrics")
#         try:
#             metric = getattr(mod_metric, self.opts.metric)
#             self.metric = dc.metrics.Metric(
#                 metric, np.mean, mode='regression')
#         except AttributeError:
#             print(f"Metric: {self.opts.metric} does not exist in deepchem")
#             raise


#     def split_data(self, dataset) -> None:
#         """
#         Split the entire dataset into a train, validate and test subsets.
#         """
#         logger.info("splitting the data into train, validate and test subsets")
#         splitter = dc.splits.ScaffoldSplitter()
#         self.data = DataSplitted(
#             *splitter.train_valid_test_split(dataset))

#     def transform_data(self):
#         """
#         Normalize the data to have zero-mean and unit-standard-deviation.
#         """
#         logger.info("Transforming the data")
#         self.transformers = [dc.trans.NormalizationTransformer(
#             transform_y=True, dataset=self.data.train)]
#         for ds in self.data:
#             for t in self.transformers:
#                 t.transform(ds)
