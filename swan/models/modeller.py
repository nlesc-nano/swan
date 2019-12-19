"""Statistical models."""
import argparse
import logging
import tempfile
from abc import abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from rdkit.Chem import AllChem
from torch import Tensor, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from swan.log_config import config_logger
from swan.models.models import select_model

from ..features.featurizer import create_molecules, generate_fingerprints
from ..input_validation import validate_input
from ..plot import create_scatter_plot
from .datasets import FingerprintsDataset, MolGraphDataset

__all__ = ["FingerprintModeller", "GraphModeller", "Modeller"]

# Starting logger
LOGGER = logging.getLogger(__name__)


def main():
    """Parse the command line arguments and call the modeller class."""
    parser = argparse.ArgumentParser(description="modeller -i input.yml")
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
        rs = predict_properties(opts)
        print(rs)


class Modeller:
    """Object to create statistical models."""

    def __init__(self, opts: dict):
        """Set up a modeler object."""
        self.opts = opts
        self.data = pd.read_csv(opts.dataset_file, index_col=0).reset_index(drop=True)
        self.data['molecules'] = create_molecules(self.data['smiles'].to_numpy())

        if opts.use_cuda and opts.mode == "train":
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.create_new_model()
        self.sanitize_data()

    def sanitize_data(self):
        """Check that the data in the DataFrame is valid."""
        # discard nan values
        self.data.dropna(inplace=True)

        # Create conformers
        self.data['molecules'].apply(lambda mol: AllChem.EmbedMolecule(mol))
        # Discard molecules that do not have conformer
        LOGGER.info("Removing moleculas that don't have any conformer.")
        self.data = self.data[self.data['molecules'].apply(lambda x: x.GetNumConformers()) >= 1]

    def create_new_model(self):
        """Configure a new model."""
        # Create an Network architecture
        self.network = select_model(self.opts)
        self.network = self.network.to(self.device)

        # select an optimizer
        optimizers = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
        config = self.opts.torch_config.optimizer
        fun = optimizers[config["name"]]
        self.optimizer = fun(self.network.parameters(), config["lr"])

        # Create loss function
        self.loss_func = nn.MSELoss()

    def load_data(self):
        """Create loaders for the train and validation dataset."""
        self.train_loader = self.create_data_loader(self.index_train)
        self.valid_loader = self.create_data_loader(self.index_valid)

    @abstractmethod
    def create_data_loader(self, indices: np.array) -> DataLoader:
        """Create a DataLoader instance for the data."""
        pass

    def train_model(self):
        """Train a statistical model."""
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

    def train_batch(self, tensor: Tensor, y_batch: Variable) -> float:
        """Train a single batch."""
        prediction = self.network(tensor)
        loss = self.loss_func(prediction, y_batch)
        loss.backward()              # backpropagation, compute gradients
        self.optimizer.step()        # apply gradients
        self.optimizer.zero_grad()   # clear gradients for next train

        cpu_tensor = loss.data.cpu()

        return cpu_tensor.numpy()

    def evaluate_model(self) -> float:
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
        LOGGER.info(f"validation loss: {mean_val_loss}")
        return mean_val_loss

    def predict(self, tensor: Tensor):
        """Use a previously trained model to predict."""
        with torch.no_grad():
            self.network.load_state_dict(torch.load(self.opts.model_path))
            self.network.eval()  # Set model to evaluation mode
            predicted = self.network(tensor)
        return predicted

    def to_numpy_detached(self, tensor: Tensor):
        """Create a view of a Numpy array in CPU."""
        tensor = tensor.cpu() if self.opts.use_cuda else tensor
        return tensor.detach().numpy()

    def split_data(self, frac: float = 0.2):
        """Split the data into a training and test set."""
        size_valid = int(self.data.index.size * frac)
        # Sample the indices without replacing
        self.index_valid = np.random.choice(self.data.index, size=size_valid, replace=False)
        self.index_train = np.setdiff1d(self.data.index, self.index_valid, assume_unique=True)

    def transform_labels(self) -> pd.DataFrame:
        """Create a new column with the transformed target."""
        self.data['transformed_labels'] = np.log(self.data[self.opts.property])


class FingerprintModeller(Modeller):
    """Object to create models using fingerprints."""

    def create_data_loader(self, indices: np.array) -> DataLoader:
        """Create a DataLoader instance for the data."""
        dataset = FingerprintsDataset(
            self.data.loc[indices], 'transformed_labels',
            self.opts.featurizer.fingerprint,
            self.opts.model.input_cells)
        return DataLoader(
            dataset=dataset, batch_size=self.opts.torch_config.batch_size)

    def plot_evaluation(self):
        """Create a scatter plot of the predict values vs the ground true."""
        dataset = self.valid_loader.dataset
        fingerprints = dataset.fingerprints.to(self.device)
        predicted = self.to_numpy_detached(self.network(fingerprints))
        expected = np.stack(dataset.labels).flatten()
        create_scatter_plot(predicted, expected)


class GraphModeller(Modeller):
    """Object to create models using molecular graphs."""

    def create_data_loader(self, indices: np.array) -> DataLoader:
        """Create a DataLoader instance for the data."""
        root = tempfile.mkdtemp(prefix="dataset_")
        dataset = MolGraphDataset(root, self.data.loc[indices], 'transformed_labels')
        return tg.data.DataLoader(
            dataset=dataset, batch_size=self.opts.torch_config.batch_size)

    def train_model(self):
        """Train a statistical model."""
        LOGGER.info("TRAINING STEP")
        # Set the model to training mode
        self.network.train()

        print("epochs: ", self.opts.torch_config.epochs)
        for epoch in range(self.opts.torch_config.epochs):
            loss_batch = 0
            for batch in self.train_loader:
                # if self.opts.use_cuda:
                #     batch = batch.to('cuda')
                loss_batch = self.train_batch(batch, batch.y)
                print("loss: ", loss_batch)
                mean = loss_batch / self.opts.torch_config.batch_size
                if epoch % self.opts.torch_config.frequency_log_epochs == 0:
                    LOGGER.info(f"Loss: {mean}")

        # Save the models
        torch.save(self.network.state_dict(), self.opts.model_path)

    def evaluate_model(self) -> float:
        """Evaluate the model against the validation dataset."""
        LOGGER.info("VALIDATION STEP")
        # Disable any gradient calculation
        with torch.no_grad():
            self.network.eval()
            val_loss = 0
            for batch in self.valid_loader:
                predicted = self.network(batch)
                val_loss += self.loss_func(batch.y, predicted)
            mean_val_loss = val_loss / self.opts.torch_config.batch_size
        LOGGER.info(f"validation loss: {mean_val_loss}")
        return mean_val_loss


def train_and_validate_model(opts: dict) -> None:
    """Train the model usign the data specificied by the user."""
    modeller = FingerprintModeller if 'fingerprint' in opts.featurizer else GraphModeller
    researcher = modeller(opts)
    researcher.transform_labels()
    researcher.split_data()
    researcher.load_data()
    researcher.train_model()
    researcher.evaluate_model()
    # researcher.plot_evaluation()


def predict_properties(opts: dict) -> Tensor:
    """Use a previous trained model to predict properties."""
    LOGGER.info(f"Loading previously trained model from: {opts.model_path}")
    researcher = FingerprintModeller(opts)
    # Generate features
    fingerprints = generate_fingerprints(
        researcher.data['molecules'], opts.featurizer.fingerprint, opts.model.input_cells)
    fingerprints = torch.from_numpy(fingerprints)
    predicted = researcher.to_numpy_detached(researcher.predict(fingerprints))
    transformed = np.exp(predicted)
    df = pd.DataFrame({'smiles': researcher.data['smiles'].to_numpy(),
                       'predicted_property': transformed.flatten()})
    return df
