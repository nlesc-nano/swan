from .input_validation import validate_input
from .metadata_models import (data_hyperparam_search, default_hyperparameters)
from collections import namedtuple
from datetime import datetime
from deepchem.models.models import Model
from deepchem.models.tensorgraph.fcnet import MultitaskRegressor
from deepchem.utils.evaluate import Evaluator
from functools import partial
from importlib import import_module
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

    # log date
    logger.info(f"Starting at:{datetime.now()}")

    # Check that the input is correct
    opts = validate_input(Path(args.i))

    # Train the model
    if opts.interface["name"].lower() == "sklearn":
        researcher = ModelerSKlearn(opts)
    else:
        researcher = ModelerTensorGraph(opts)

    # train the model
    if opts.load_model:
        model = researcher.load_model()
    else:
        model = researcher.train_model()

    # Check how good is the model
    researcher.evaluate_model(model)

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
        # Import the metric
        mod_metric = import_module("deepchem.metrics")
        try:
            metric = getattr(mod_metric, self.opts.metric)
            self.metric = dc.metrics.Metric(
                metric, np.mean, mode='regression')
        except AttributeError:
            print(f"Metric: {self.opts.metric} does not exist in deepchem")
            raise

    def load_data(self):
        """
        Load a dataset
        """
        logger.info(f"Loading data from {self.opts.dataset_file}")
        print("file name: ", self.opts.dataset_file)
        if Path(self.opts.dataset_file).suffix != ".csv":
            dataset = dc.utils.save.load_from_disk(self.opts.dataset_file)

        else:
            loader = dc.data.CSVLoader(tasks=self.opts.tasks, smiles_field="smiles",
                                       featurizer=self.featurizer)
            dataset = loader.featurize(self.opts.dataset_file)

        if self.opts.save_dataset:
            file_name = Path(self.opts.workdir) / \
                f"{self.opts.filename_to_store_dataset}.joblib"
            logger.info(f"saving dataset to: {file_name}")
            dc.utils.save.save_to_disk(dataset, file_name)

        return dataset

    def load_model(self) -> Model:
        """
        Load model from disk
        """
        dataset = self.load_data()
        self.split_data(dataset)

        model = self.select_model()

        return model.load_from_dir(self.opts.model_dir)

    def split_data(self, dataset) -> None:
        """
        Split the entire dataset into a train, validate and test subsets.
        """
        logger.info("splitting the data into train, validate and test subsets")
        splitter = dc.splits.ScaffoldSplitter()
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
        logging.info(f"Score of the model is: {score}")
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
            model_class = partial(_model_builder_tensorgraph,
                                  self.n_tasks, self.n_features)

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
            'multitaskregressor': MultitaskRegressor,
        }
        self.n_tasks = len(self.opts.tasks)
        self.n_features = self.featurizer.size

    def select_model(self) -> Model:
        """
        Return the Model object specificied by the user
        """
        model_name = self.opts.interface["model"]
        logger.info(f"Train the model using {model_name}")

        # Select model and fit it
        hyper = self.select_hyperparameters()
        hyper["model_dir"] = self.opts.model_dir

        # Create model
        tensorgraph_model = self.available_models[model_name]
        args = self._get_positional_args()
        return tensorgraph_model(*args, **hyper)

    def _get_positional_args(self) -> list:
        """
        Select the positional arguments for a given model
        """
        model_name = self.opts.interface["model"]
        dict_args = {
            'multitaskregressor': [self.n_tasks, self.n_features]
        }

        return dict_args[model_name]

    def fit_model(self) -> Model:
        """
        Fit the statistical model using the given hyperparameters or the default
        """
        model = self.select_model()
        model.fit(self.data.train, nb_epoch=self.opts.interface["epochs"])

        # save model data
        model.save()

        return model


class ModelerSKlearn(Modeler):

    def __init__(self, opts: dict):
        super().__init__(opts)
        self.available_models = {
            'randomforest': RandomForestRegressor,
            'svr': SVR,
            'kernelridge': KernelRidge,
            'bagging': BaggingRegressor}

    def select_model(self) -> Model:
        """
        Return the Model object specificied by the user
        """
        model_name = self.opts.interface["model"]
        logger.info(f"Train the model using {model_name}")

        # Select hyperparameters
        hyper = self.select_hyperparameters()
        sklearn_model = self.available_models[model_name](**hyper)
        return dc.models.SklearnModel(sklearn_model)

    def fit_model(self) -> Model:
        """
        Fit the statistical model using the given hyperparameters or the default
        """
        model = self.select_model()
        model.fit(self.data.train)

        # save the model optimized parameters
        model.save()

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
