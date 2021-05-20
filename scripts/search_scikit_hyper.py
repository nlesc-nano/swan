#!/usr/bin/env python

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import gaussian_process, svm, tree
from sklearn.model_selection import GridSearchCV
import sklearn.gaussian_process.kernels as gp

from swan.dataset import FingerprintsData
from swan.utils.log_config import configure_logger

configure_logger(Path("."))

# Starting logger
LOGGER = logging.getLogger(__name__)

path_data = Path("data/Carboxylic_acids/CDFT/cdft_random_500.csv")
properties = [
    # "Dissocation energy (nucleofuge)",
    # "Dissociation energy (electrofuge)",
    # "Electroaccepting power(w+)",
    # "Electrodonating power (w-)",
    # "Electronegativity (chi=-mu)",
    # "Electronic chemical potential (mu)",
    # "Electronic chemical potential (mu+)",
    # "Electronic chemical potential (mu-)",
    # "Electrophilicity index (w=omega)",
    # "Global Dual Descriptor Deltaf+",
    # "Global Dual Descriptor Deltaf-",
    # "Hardness (eta)",
    # "Hyperhardness (gamma)",
    # "Net Electrophilicity",
    "Softness (S)"
]


supported_models = {
    "tree": tree.DecisionTreeRegressor,
    "svm": svm.SVR,
    "gaussian": gaussian_process.GaussianProcessRegressor
}

supported_parameters = {
    "tree": {
        "criterion": ("mse", "friedman_mse", "mae"),
        "splitter": ("best", "random"),
        "max_features": ("auto", "sqrt", "log2"),
    },
    "svm": {
        "kernel": ("linear", "poly", "rbf", "sigmoid"),
        "gamma": ("scale", "auto"),
        "C": [1, 5, 10],
        "shrinking": (True, False)
    },
    "gaussian": {
        "kernel": [
            gp.ConstantKernel(1.0, (1e-1, 1e3)) * gp.RBF(10.0, (1e-3, 1e3)),
            gp.ConstantKernel(1.0, (1e-1, 1e3)) * gp.DotProduct(),
            gp.ConstantKernel(1.0, (1e-1, 1e3)) * gp.Matern(10.0, (1e-3, 1e3)),
            gp.ConstantKernel(1.0, (1e-1, 1e3)) * gp.RationalQuadratic(),
        ],
    }
}


def get_data(size: Optional[int]):
    """Get the fingerprints data."""
    # Fingerprints
    data = FingerprintsData(
        path_data, properties=properties, sanitize=False)
    data.scale_labels()
    # Take a sample
    size = len(data.fingerprints) if size is None else int(size)
    indices = np.random.choice(np.arange(len(data.fingerprints)), size=size, replace=False)
    return data.fingerprints[indices], data.labels[indices]


def search_for_hyperparameters(model_name: str, nsamples: Optional[int]):
    """Use a Grid Search for the best hyperparameters."""
    fingerprints, labels = get_data(nsamples)
    model = supported_models[model_name]()
    parameters = supported_parameters[model_name]
    grid = GridSearchCV(model, parameters, scoring="r2")
    grid.fit(fingerprints, labels.flatten())
    df = pd.DataFrame(grid.cv_results_)
    df.sort_values('rank_test_score', inplace=True)
    columns = ['params', 'mean_test_score', 'rank_test_score']
    df.to_csv(f"{model_name}_hyperparameters.csv", columns=columns, index=False)
    print(df[columns][:5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=["tree", "svm", "gaussian"], default="tree")
    parser.add_argument("-n", "--nsamples", help="Number of sample to use", default=None)
    args = parser.parse_args()

    search_for_hyperparameters(args.model, args.nsamples)


if __name__ == "__main__":
    main()
