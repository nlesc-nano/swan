"""
SCScore model taken from: https://github.com/connorcoley/scscore.
"""

import gzip
import json
import logging

import numpy as np
import pkg_resources


__all__ = ["SCScorer"]


logger = logging.getLogger(__name__)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def get_model_data(name: str) -> str:
    """look for the data for a given model with `name`."""
    path = f"data/scscore/full_reaxys_model_{name}/model.ckpt-10654.as_numpy.json.gz"
    return pkg_resources.resource_filename("swan", path)


class SCScorer():
    """Load a pretrained SCScore model and makes predictions with it.

    The model details can be found at: https://pubs.acs.org/doi/10.1021/acs.jcim.7b00622

    The model consist of:
        * input layer of either 1024 or 2048 nodes
        * five hidden layers of 300 nodes each, ReLU activation with bias
        * Single sigmoid out
        * Linear scale (1, 5)
    """

    def __init__(self, model_name: str, score_scale: float = 5.0):
        self.score_scale = score_scale
        self._load_pretrained_model(model_name)

    def _load_pretrained_model(self, model_name: str):
        """Load the models weights and biases from swan/data/scscore."""
        weight_path = get_model_data(model_name)
        logger.info(f"Loading model data from: {weight_path}")
        self._load_vars(weight_path)

    def compute_score(self, x: np.ndarray) -> np.float32:
        """Use the precomputed model to predict an score.

        Parameters
        ----------
        x
          Finger prints in matrix format (nmolecules, nbits)

        Returns
        -------
          Scores for the given fingerprints
        """
        for w, b in zip(self.weights, self.biases):
            x = np.matmul(x, w) + b
            # Apply the ReLU in all the layers except the last one
            if b.shape[0] != 1:
                x *= (x > 0)  # ReLU

        result = 1 + (self.score_scale - 1) * sigmoid(x)
        return result.flatten()

    def _load_vars(self, weight_path: str):
        """
        Load the neural network weights and biases.

        The weights and biases are stored as gz compressed json files.
        """
        with gzip.GzipFile(weight_path, 'r') as fin:
            json_bytes = fin.read()  # as UTF-8

        variables = json.loads(json_bytes.decode('utf-8'))
        self.weights = tuple(np.array(x) for x in variables[0::2])
        self.biases = tuple(np.array(x) for x in variables[1::2])
