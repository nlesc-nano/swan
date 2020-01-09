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
    elif args.mode == "cross":
        cross_validate(opts)
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
        LOGGER.info("Removing molecules that don't have any conformer.")
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

        previous_loss = 0  # Check if optimization reaches a plateau
        for epoch in range(self.opts.torch_config.epochs):
            loss_all = 0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss_all += self.train_batch(x_batch, y_batch) * len(x_batch)
            relative_loss = loss_all / len(self.index_train)
            if abs(previous_loss - relative_loss) < 1e-4:
                break  # A plateau has been reached
            previous_loss = relative_loss

            LOGGER.info(f"Loss: {relative_loss / len(self.train_loader)}")

        # Save the models
        torch.save(self.network.state_dict(), self.opts.model_path)

    def train_batch(self, tensor: Tensor, y_batch: Variable) -> float:
        """Train a single batch."""
        prediction = self.network(tensor)
        loss = self.loss_func(prediction, y_batch)
        loss.backward()              # backpropagation, compute gradients
        self.optimizer.step()        # apply gradients
        self.optimizer.zero_grad()   # clear gradients for next train

        return loss.item()

    def evaluate_model(self) -> float:
        """Evaluate the model against the validation dataset."""
        LOGGER.info("VALIDATION STEP")
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
            LOGGER.info(f"Loss: {loss_all / len(self.valid_loader)}")
        return torch.cat(results), torch.cat(expected)

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
        size_valid = int(len(self.data.index) * frac)
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

        previous_loss = 0  # Check if optimization reaches a plateau
        for epoch in range(self.opts.torch_config.epochs):
            loss_all = 0
            for batch in self.train_loader:
                batch.to(self.device)
                loss_batch = self.train_batch(batch, batch.y)
                loss_all += batch.num_graphs * loss_batch
            relative_loss = loss_all / len(self.index_train)
            if abs(previous_loss - relative_loss) < 1e-4:
                break  # A plateau has been reached
            previous_loss = relative_loss

            LOGGER.info(f"Loss: {relative_loss}")

        # Save the models
        torch.save(self.network.state_dict(), self.opts.model_path)

    def evaluate_model(self) -> float:
        """Evaluate the model against the validation dataset."""
        LOGGER.info("VALIDATION STEP")
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
            LOGGER.info(f"Loss: {loss_all / len(self.index_valid)}")
            print("validation loss: ", loss_all / len(self.index_valid))

        return torch.cat(results), torch.cat(expected)


def train_and_validate_model(opts: dict) -> None:
    """Train the model usign the data specificied by the user."""
    modeller = FingerprintModeller if 'fingerprint' in opts.featurizer else GraphModeller
    researcher = modeller(opts)
    researcher.transform_labels()
    researcher.split_data()
    researcher.load_data()
    researcher.train_model()
    predicted, expected = researcher.evaluate_model()
    create_scatter_plot(*[researcher.to_numpy_detached(x) for x in (predicted, expected)])


def predict_properties(opts: dict) -> Tensor:
    """Use a previous trained model to predict properties."""
    LOGGER.info(f"Loading previously trained model from: {opts.model_path}")
    modeller = FingerprintModeller if 'fingerprint' in opts.featurizer else GraphModeller
    researcher = modeller(opts)
    # Generate features
    if 'fingerprint' in opts.featurizer:
        features = generate_fingerprints(
            researcher.data['molecules'], opts.featurizer.fingerprint, opts.model.input_cells)
        features = torch.from_numpy(features).to(researcher.device)
    else:
        # Create a single minibatch with the data to predict
        dataset = MolGraphDataset(tempfile.mkdtemp(prefix="dataset_"), researcher.data)
        data_loader = tg.data.DataLoader(
            dataset=dataset, batch_size=len(researcher.data['molecules']))
        features = next(iter(data_loader))
        features.to(researcher.device)

    # Predict the property value and report
    predicted = researcher.to_numpy_detached(researcher.predict(features))
    transformed = np.exp(predicted)
    df = pd.DataFrame({'smiles': researcher.data['smiles'].to_numpy(),
                       'predicted_property': transformed.flatten()})
    return df


def cross_validate(opts: dict) -> Tensor:
    """Run a cross validation with the given `opts`."""
    pass
