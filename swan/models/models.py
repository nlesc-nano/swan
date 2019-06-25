from .input_validation import validate_input
from .metadata_models import data_hyperparam_search
from collections import namedtuple
from deepchem.models.models import Model
from deepchem.utils.evaluate import Evaluator
from functools import partial
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from swan.log_config import config_logger

import argparse
import logging
import deepchem as dc

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
            print("best_model_hyperparams: ", best_model_hyperparams)
        # # Use the random forest approach
        # model = self.fit_model()

        return best_model

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
        metric = dc.metrics.Metric(dc.metrics.r2_score)
        evaluator = Evaluator(model, self.data.valid, self.transformers)
        score = evaluator.compute_model_performance([metric])
        print("score: ", score)

    def fit_model(self):
        """
        Fit the statistical model
        """
        available_models = {
            'randomforest': call_random_forest,
            'svr': call_SVR}
        model_name = self.opts.interface["model"]

        return available_models[model_name](self.data.train)

    def optimize_hyperparameters(self) -> tuple:
        """
        Search for the best hyperparameters for a given model
        """
        models = {'randomforest': RandomForestRegressor}
        model_name = self.opts.interface["model"]
        regressor = models[model_name]

        builder = partial(_model_builder, regressor)

        # optimizer = dc.hyper.HyperparamOpt(_random_forest_model_builder)
        optimizer = dc.hyper.HyperparamOpt(builder)
        params_dict = data_hyperparam_search[model_name]
        best_model, best_model_hyperparams, all_models_results = optimizer.hyperparam_search(
            params_dict, self.data.train, self.data.valid, self.transformers,
            metric=self.metric)

        return best_model, best_model_hyperparams, all_models_results


def call_sklearn_model(train, fun: callable):
    """
    Call a SKlearn model given by `fun` using `train` data.
    """
    model = dc.models.SklearnModel(fun)
    model.fit(train)

    return model


def call_random_forest(train):
    """
    Call the sklearn `RandomForestRegressor`
    """
    logger.info("Train the model using a random forest regression")
    sklearn_model = RandomForestRegressor(
        n_estimators=100, max_features='sqrt', n_jobs=-1)
    return call_sklearn_model(train, sklearn_model)


def search_best_random_forest(self) -> tuple:
    """
    Search brute force for the best random forest model
    """
    params_dict = {
        "n_estimators": [10, 100],
        "max_features": ["auto", "sqrt", "log2", None]
    }
    metric = dc.metrics.Metric(dc.metrics.r2_score)
    optimizer = dc.hyper.HyperparamOpt(_random_forest_model_builder)
    best_rf, best_rf_hyperparams, all_rf_results = optimizer.hyperparam_search(
        params_dict, self.data.train, self.data.valid, self.transformers,
        metric=metric)


def _random_forest_model_builder(model_params, model_dir="."):
    """
    Search for the best parameters to build a random forest statistical
    model
    """
    sklearn_model = RandomForestRegressor(**model_params)
    return dc.models.SklearnModel(sklearn_model, model_dir)


def _model_builder(regressor: object, model_params: dict, model_dir: str = "."):
    """
    Create a SklearnModel
    """
    sklearn_model = regressor(**model_params)
    return dc.models.SklearnModel(sklearn_model, model_dir)


def call_SVR(train):
    """
    call the support vector regression
    """
    logger.info("Train the model using a support vector regression")
    sklearn_model = SVR()
    return call_sklearn_model(train, sklearn_model)
