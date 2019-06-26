from .input_validation import validate_input
from .metadata_models import (data_hyperparam_search, default_hyperparameters)
from collections import namedtuple
from deepchem.models.models import Model
from deepchem.utils.evaluate import Evaluator
from functools import partial
from pathlib import Path
from sklearn.ensemble import (BaggingRegressor, RandomForestRegressor)
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from swan.log_config import config_logger

import argparse
import logging
import deepchem as dc

__all__ = ["Modeler"]

# Starting logger
logger = logging.getLogger(__name__)

DataSplitted = namedtuple('DataSplitted', 'train, valid, test')


def main():
    parser = argparse.ArgumentParser(description="train -i input.yml")
    # configure logger
    parser.add_argument('-i', required=True,
                        help="Input file with options")
    parser.add_argument('-w', help="workdir", default=Path("."))
    args = parser.parse_args()

    # start logger
    config_logger(args.w)

    # Check that the input is correct
    opts = validate_input(Path(args.i))

    # Train the model
    researcher = Modeler(opts)
    model = researcher.train_model()

    # Check how good is the model
    researcher.evaluate_model(model)

    # predict
    model.predict(researcher.data.test)


class Modeler:

    def __init__(self, opts: dict):
        self.opts = opts
        self.available_models = {
            'randomforest': RandomForestRegressor,
            'svr': SVR,
            'kernelridge': KernelRidge,
            'bagging': BaggingRegressor}

        self.create_metric()

    def create_metric(self) -> None:
        """
        Create instances of the metric to use
        """
        if self.opts.metric == 'r2_score':
            self.metric = dc.metrics.Metric(dc.metrics.r2_score)
        else:
            msg = f"Metric: {self.opts.metric} has not been implemented"
            raise NotImplementedError(msg)

    def train_model(self) -> Model:
        """
        Use the data and `options` provided by the user to create an statistical
        model.
        """
        # Use CircularFingerprint for featurization
        featurizer = dc.feat.CircularFingerprint(size=1024)

        # Load data
        logger.info("Loading data")
        loader = dc.data.CSVLoader(tasks=self.opts.tasks, smiles_field="smiles",
                                   featurizer=featurizer)
        dataset = loader.featurize(self.opts.csv_file)

        # Split the data into train/validation/test sets
        self.split_data(dataset)

        # Normalize the data
        self.transform_data()

        # Optimize hyperparameters
        if self.opts["optimize_hyperparameters"]:
            best_model, best_model_hyperparams, all_models_results = self.optimize_hyperparameters()
            return best_model
        else:
            # Use the statistical model as it is
            return self.fit_model()

    def split_data(self, dataset) -> None:
        """
        Split the entire dataset into a train, validate and test subsets.
        """
        splitter = dc.splits.ScaffoldSplitter(self.opts.csv_file)
        self.data = DataSplitted(
            *splitter.train_valid_test_split(dataset))

    def transform_data(self):
        """
        Normalize the data to have zero-mean and unit-standard-deviation.
        """
        self.transformers = [dc.trans.NormalizationTransformer(
            transform_y=True, dataset=self.data.train)]
        for ds in self.data:
            for t in self.transformers:
                t.transform(ds)

    def evaluate_model(self, model) -> None:
        """
        Evaluate the predictive power of the model
        """
        evaluator = Evaluator(model, self.data.valid, self.transformers)
        score = evaluator.compute_model_performance([self.metric])
        print("score: ", score)

    def fit_model(self) -> Model:
        """
        Fit the statistical model using the given parameters or the default
        """
        model_name = self.opts.interface["model"]
        logger.info(f"Train the model using {model_name}")

        # Use the parameters provided by the user or the defaults
        if not self.opts.interface["parameters"]:
            parameters = default_hyperparameters.get(model_name, {})
        else:
            parameters = self.opts.interface["parameters"]

        # Select model and fit it
        sklearn_model = self.available_models[model_name](**parameters)
        model = dc.models.SklearnModel(sklearn_model)
        model.fit(self.data.train)

        return model

    def optimize_hyperparameters(self) -> tuple:
        """
        Search for the best hyperparameters for a given model
        """
        model_name = self.opts.interface["model"]
        regressor = self.available_models[model_name]

        builder = partial(_model_builder, regressor)

        # optimizer = dc.hyper.HyperparamOpt(_random_forest_model_builder)
        optimizer = dc.hyper.HyperparamOpt(builder)
        params_dict = data_hyperparam_search[model_name]
        best_model, best_model_hyperparams, all_models_results = optimizer.hyperparam_search(
            params_dict, self.data.train, self.data.valid, self.transformers,
            metric=self.metric)

        return best_model, best_model_hyperparams, all_models_results


def _model_builder(regressor: object, model_params: dict, model_dir: str = "."):
    """
    Create a SklearnModel
    """
    sklearn_model = regressor(**model_params)
    return dc.models.SklearnModel(sklearn_model, model_dir)


def call_sklearn_model(train, model_name: str, sklearn_model: object):
    """
    Call a SKlearn model given by `fun` using `train` data.
    """
    logger.info(f"Train the model using {model_name}")
    model = dc.models.SklearnModel(sklearn_model)
    model.fit(train)

    return model
