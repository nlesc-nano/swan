"""Module to create statistical models using scikit learn."""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from sklearn import gaussian_process, svm, tree

from ..dataset.fingerprints_data import FingerprintsData

PathLike = Union[str, Path]

LOGGER = logging.getLogger(__name__)


class SKModeller:
    """Create statistical models using the scikit learn library."""

    def __init__(self, data: FingerprintsData, name: str, **kwargs):
        """Class constructor.

        Parameters
        ----------
        data
            FingerprintsData object containing the dataset
        name
            scikit learn model to use
        """
        self.fingerprints = data.fingerprints.numpy()
        self.labels = data.dataset.labels.numpy()
        self.path_model = "swan_skmodeller.pkl"

        supported_models = {
            "tree": tree.DecisionTreeRegressor,
            "svm": svm.SVR,
            "gaussian": gaussian_process.GaussianProcessRegressor
        }

        if name.lower() in supported_models:
            self.model = supported_models[name.lower()](**kwargs)
        else:
            raise RuntimeError(f"There is not model name: {name}")

        LOGGER.info(f"Created {name} model")

    def split_data(self, frac: Tuple[float, float]):
        """Split the data into a training and validation set.

        Parameters
        ----------
        frac
            fraction to divide the dataset, by default [0.8, 0.2]
        """
        # Generate random indices to train and validate the model
        size = len(self.fingerprints)
        indices = np.arange(size)
        np.random.shuffle(indices)

        ntrain = int(size * frac[0])
        self.features_trainset = self.fingerprints[indices[:ntrain]]
        self.features_validset = self.fingerprints[indices[ntrain:]]
        self.labels_trainset = self.labels[indices[:ntrain]]
        self.labels_validset = self.labels[indices[ntrain:]]

    def train_model(self, frac: Tuple[float, float] = (0.8, 0.2)):
        """Train the model using the given data.

        Parameters
        ----------
        frac
            fraction to divide the dataset, by default [0.8, 0.2]
        """
        self.split_data(frac)
        self.model.fit(self.features_trainset, self.labels_trainset)
        self.save_model()

    def save_model(self):
        """Store the trained model."""
        with open(self.path_model, 'wb') as handler:
            pickle.dump(self.model, handler)

    def validate_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """Check the model prediction power."""
        predicted = self.model.predict(self.features_validset)
        expected = self.labels_validset
        score = self.model.score(self.features_validset, expected)
        LOGGER.info(f"Validation R^2 score: {score}")
        return predicted, expected

    def load_model(self, path_model: Optional[PathLike]) -> None:
        """Load the model from the state file."""
        path_model = self.path_model if path_model is None else path_model
        with open(path_model, 'rb') as handler:
            self.model = pickle.load(handler)

    def predict(self, inp_data: np.ndarray) -> np.ndarray:
        """Used the previously trained model to predict properties.

        Parameters
        ----------
        inp_data
            Matrix containing a given fingerprint for each row

        Returns
        -------
        Array containing the predicted results
        """
        return self.model.predict(inp_data)
