"""
SCScore model taken from: https://github.com/connorcoley/scscore.
"""

import gzip
import json
import logging
import math

import numpy as np
import pkg_resources

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def get_model_data(name: str) -> str:
    """look for the data for a given model with `name`."""
    path = f"data/scscore/full_reaxys_model_{name}/model.ckpt-10654.as_numpy.json.gz"
    return pkg_resources.resource_filename("swan", path)


class SCScorer():
    """Load a pretrained SCScore model and makes predictions with it.

    The model details can be found at: https://pubs.acs.org/doi/10.1021/acs.jcim.7b00622
    """

    def __init__(self, model_name: str, score_scale: int = 5.0):
        self.score_scale = score_scale
        self.load_pretrained_model(model_name)

    def load_pretrained_model(self, model_name: str):
        """Load the models weights and biases from swan/data/scscore."""
        weight_path = get_model_data(model_name)
        logger.info(f"Loading model data from: {weight_path}")
        self._load_vars(weight_path)

    def compute_score(self, x: np.ndarray) -> np.float32:
        # Each pair of vars is a weight and bias term
        npairs = len(self.vars)
        for i in range(0, npairs, 2):
            last_layer = (i == npairs - 2)
            w = self.vars[i]
            b = self.vars[i + 1]
            x = np.matmul(x, w) + b
            if not last_layer:
                x = x * (x > 0)  # ReLU
        x = 1 + (self.score_scale - 1) * sigmoid(x)
        return x

    def _load_vars(self, weight_path):
        with gzip.GzipFile(weight_path, 'r') as fin:
            json_bytes = fin.read()  # as UTF-8

        variables = json.loads(json_bytes.decode('utf-8'))
        self.vars = [np.array(x) for x in variables]

# 1.4323 <--- CCCOCCC
# 1.8913 <--- CCCNc1ccccc1
# 1.3429 <--- CCCOCCC
# 1.8087 <--- CCCNc1ccccc1
