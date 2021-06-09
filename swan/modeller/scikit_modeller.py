"""Module to create statistical models using scikit learn."""

import logging
import pickle
from typing import Optional, Tuple

import numpy as np
from sklearn import gaussian_process, svm, tree

from ..dataset.fingerprints_data import FingerprintsData
from ..type_hints import PathLike
from .base_modeller import BaseModeller

LOGGER = logging.getLogger(__name__)


class SKModeller(BaseModeller[np.ndarray]):
    """Create statistical models using the scikit learn library."""

    def __init__(self, name: str, data: FingerprintsData, replace_state: bool = False, **kwargs):
        """Class constructor.

        Parameters
        ----------
        name
            scikit learn model to use
        data
            FingerprintsData object containing the dataset
        replace_state
            Remove previous state file
        """
        super(SKModeller, self).__init__(data, replace_state)
        self.fingerprints = data.fingerprints.numpy()
        self.labels = data.dataset.labels.numpy()
        self.path_model = "swan_skmodeller.pkl"

        supported_models = {
            "decision_tree": tree.DecisionTreeRegressor,
            "svm": svm.SVR,
            "gaussian_process": gaussian_process.GaussianProcessRegressor
        }

        if name.lower() in supported_models:
            self.model = supported_models[name.lower()](**kwargs)
        else:
            raise RuntimeError(f"There is not model name: {name}")

        LOGGER.info(f"Created {name} model")

    def train_model(self, frac: Tuple[float, float] = (0.8, 0.2)):
        """Train the model using the given data.

        Parameters
        ----------
        frac
            fraction to divide the dataset, by default [0.8, 0.2]
        """
        self.split_fingerprint_data(frac)
        self.model.fit(self.features_trainset, self.labels_trainset.flatten())
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
