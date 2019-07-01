from .input_validation import validate_input
from .metadata_models import (data_hyperparam_search, default_hyperparameters)
from collections import namedtuple
from deepchem.models.models import Model
from deepchem.models.tensorgraph.fcnet import MultitaskRegressor
from deepchem.utils.evaluate import Evaluator
from functools import partial
from pathlib import Path
from sklearn.ensemble import (BaggingRegressor, RandomForestRegressor)
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from swan.log_config import config_logger

import argparse
import deepchem as dc
import logging
import numpy as np

__all__ = ["Modeler", "ModelerSKlearn", "ModelerTensorGraph"]

# Starting logger
logger = logging.getLogger(__name__)

DataSplitted = namedtuple('DataSplitted', 'train, valid, test')


def main():
    parser = argparse.ArgumentParser(description="train -i input.yml")
    # configure logger
    parser.add_argument('-i', required=True,
                        help="Input file with options")
    parser.add_argument('-w', help="workdir", default=".")
    args = parser.parse_args()

    # start logger
    config_logger(Path(args.w))

    # Check that the input is correct
    opts = validate_input(Path(args.i))

    # Train the model
    if opts.interface["name"].lower() == "sklearn":
        researcher = ModelerSKlearn(opts)
    else:
        researcher = ModelerTensorGraph(opts)

    # train the model
    model = researcher.train_model()

    # Check how good is the model
    researcher.evaluate_model(model)

    if opts.save:
        model.save()

    # # predict
    # model.predict(researcher.data.test)


class Modeler:

    def __init__(self, opts: dict):
        self.opts = opts

        # Use CircularFingerprint for featurization
        self.featurizer = dc.feat.CircularFingerprint(size=2048)

        self.select_metric()

    def select_metric(self) -> None:
        """
        Create instances of the metric to use
        """
        if self.opts.metric == 'r2_score':
            self.metric = dc.metrics.Metric(
                dc.metrics.r2_score, np.mean, mode='regression')
        else:
            msg = f"Metric: {self.opts.metric} has not been implemented"
            raise NotImplementedError(msg)

    def load_data(self):
        """
        Load a dataset
        """
        logger.info("Loading data")
        loader = dc.data.CSVLoader(tasks=self.opts.tasks, smiles_field="smiles",
                                   featurizer=self.featurizer)
        return loader.featurize(self.opts.csv_file)

    def split_data(self, dataset) -> None:
        """
        Split the entire dataset into a train, validate and test subsets.
        """
        logger.info("splitting the data into train, validate and test subsets")
        splitter = dc.splits.ScaffoldSplitter(self.opts.csv_file)
        self.data = DataSplitted(
            *splitter.train_valid_test_split(dataset))

    def transform_data(self):
        """
        Normalize the data to have zero-mean and unit-standard-deviation.
        """
        logger.info("Transforming the data")
        self.transformers = [dc.trans.NormalizationTransformer(
            transform_y=True, dataset=self.data.train)]
        for ds in self.data:
            for t in self.transformers:
                t.transform(ds)

    def train_model(self) -> Model:
        """
        Use the data and `options` provided by the user to create an statistical
        model.
        """
        # Load the data from a csv file
        dataset = self.load_data()

        # Split the data into train/validation/test sets
        self.split_data(dataset)

        # Normalize the data
        self.transform_data()

        # Optimize hyperparameters
        if self.opts["optimize_hyperparameters"]:
            best_model, best_model_hyperparams, _ = self.optimize_hyperparameters()
            logger.info(f"best hyperparameters: {best_model_hyperparams}")
            return best_model
        else:
            # Use the statistical model as it is
            return self.fit_model()

    def evaluate_model(self, model) -> None:
        """
        Evaluate the predictive power of the model
        """
        evaluator = Evaluator(model, self.data.valid, self.transformers)
        score = evaluator.compute_model_performance([self.metric])
        print("score: ", score)

    def select_hyperparameters(self) -> dict:
        """
        Use the parameters provided by the user or the defaults
        """
        model_name = self.opts.interface["model"]
        if not self.opts.interface["parameters"]:
            return default_hyperparameters.get(model_name, {})
        else:
            return self.opts.interface["parameters"]

    def optimize_hyperparameters(self) -> tuple:
        """
        Search for the best hyperparameters for a given model
        """
        model_name = self.opts.interface["model"]

        if self.opts.interface["name"].lower() == "sklearn":
            regressor = self.available_models[model_name]
            model_class = partial(_model_builder_sklearn, regressor)
        else:
            model_class = partial(_model_builder_tensorgraph, self.n_tasks, self.n_features)

        optimizer = dc.hyper.HyperparamOpt(model_class)
        params_dict = data_hyperparam_search[model_name]
        best_model, best_model_hyperparams, all_models_results = optimizer.hyperparam_search(
            params_dict, self.data.train, self.data.valid, self.transformers,
            metric=self.metric)

        return best_model, best_model_hyperparams, all_models_results


class ModelerTensorGraph(Modeler):

    def __init__(self, opts: dict):
        super().__init__(opts)
        self.available_models = {
            'multitaskregressor': MultitaskRegressor
        }
        self.n_tasks = len(self.opts.tasks)
        self.n_features = self.featurizer.size

    def fit_model(self) -> Model:
        """
        Fit the statistical model using the given hyperparameters or the default
        """
        model_name = self.opts.interface["model"]
        logger.info(f"Train the model using {model_name}")

        # Select model and fit it
        hyper = self.select_hyperparameters()

        # Create model
        tensorgraph_model = self.available_models[model_name]
        model = tensorgraph_model(self.n_tasks, self.n_features, **hyper)

        model.fit(self.data.train, nb_epoch=self.opts.interface["epochs"])

        return model


class ModelerSKlearn(Modeler):

    def __init__(self, opts: dict):
        super().__init__(opts)
        self.available_models = {
            'randomforest': RandomForestRegressor,
            'svr': SVR,
            'kernelridge': KernelRidge,
            'bagging': BaggingRegressor}

    def fit_model(self) -> Model:
        """
        Fit the statistical model using the given hyperparameters or the default
        """
        model_name = self.opts.interface["model"]
        logger.info(f"Train the model using {model_name}")

        # Select model and fit it
        hyper = self.select_hyperparameters()
        sklearn_model = self.available_models[model_name](**hyper)
        model = dc.models.SklearnModel(sklearn_model)
        model.fit(self.data.train)

        return model


def _model_builder_tensorgraph(n_tasks: int, n_features: int, model_params: dict,
                               model_dir: str = "."):
    """
    Create a TensorGraph Model
    """
    return MultitaskRegressor(n_tasks, n_features, **model_params)


def _model_builder_sklearn(regressor: object, model_params: dict, model_dir: str = "."):
    """
    Create a Sklearn Model
    """
    sklearn_model = regressor(**model_params)
    return dc.models.SklearnModel(sklearn_model, model_dir)
