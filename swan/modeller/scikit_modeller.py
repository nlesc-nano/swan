"""Module to create statistical models using scikit learn."""

import logging
import pickle
from typing import Optional, Tuple

import numpy as np
from sklearn import gaussian_process, svm, tree

from ..dataset.fingerprints_data import FingerprintsData
from ..dataset.splitter import split_dataset
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

    def train_model(self, frac: Tuple[float, float] = (0.8, 0.2)) -> None:
        """Train the model using the given data.

        Parameters
        ----------
        frac
            fraction to divide the dataset, by default [0.8, 0.2]
        """
        self.split_data(frac)
        self.model.fit(self.features_trainset, self.labels_trainset.flatten())
        self.save_model()

    def split_data(self, frac: Tuple[float, float]) -> None:
        """Split the dataset into a training and validation set."""
        partition = split_dataset(self.fingerprints, self.labels, frac)
        self.features_trainset = partition.features_trainset
        self.features_validset = partition.features_validset
        self.labels_trainset = partition.labels_trainset
        self.labels_validset = partition.labels_validset

        # Split the smiles using the same partition than the features
        indices = partition.indices
        ntrain = partition.ntrain
        self.state.store_array("smiles_train", self.smiles[indices[:ntrain]], dtype="str")
        self.state.store_array("smiles_validate", self.smiles[indices[ntrain:]], dtype="str")

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
        return tuple(self.inverse_transform(x) for x in (predicted, expected))

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

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """Unscale ``arr`` using the fitted scaler.

        Parameters
        ----------
        arr
            Array to inverse-transform

        Returns
        -------
        Inverse-Transformed array
        """
        def invert(arr: np.ndarray) -> np.ndarray:
            if len(arr.shape) == 1:
                arr = arr.reshape(-1, 1)

            return arr

        return self.data.transformer.inverse_transform(invert(arr))
