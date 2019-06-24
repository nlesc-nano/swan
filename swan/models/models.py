from .input_validation import validate_input
from collections import namedtuple
from deepchem.utils.evaluate import Evaluator
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
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

    def train_model(self):
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

        # Use the random forest approach
        model = self.fit_model()

        return model

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

    def evaluate_model(self, model):
        """
        Evaluate the predictive power of the model
        """
        metric = dc.metrics.Metric(dc.metrics.r2_score)
        evaluator = Evaluator(model, self.data.valid, self.transformers)
        score = evaluator.compute_model_performance([metric])
        print("score: ", score)

    def search_best_random_forest(self):
        """
        Search brute force for the best random forest model
        """
        params_dict = {
            "n_estimators": [10, 100, 1000],
            "max_features": ["auto", "sqrt", "log2", None]
        }
        metric = dc.metrics.Metric(dc.metrics.r2_score)
        optimizer = dc.hyper.HyperparamOpt(_random_forest_model_builder)
        best_rf, best_rf_hyperparams, all_rf_results = optimizer.hyperparam_search(
            params_dict, self.data.train, self.data.valid, self.transformers,
            metric=metric)

    def fit_model(self):
        """
        Fit the statistical model
        """
        available_models = {'randomforest': call_random_forest}
        model_name = self.opts.interface["model"]

        return available_models[model_name](self.data.train)


def call_random_forest(train):
    """
    Call the sklearn `RandomForestRegressor`
    """
    logger.info("Train the model using a random forest")
    sklearn_model = RandomForestRegressor(n_estimators=100, max_features='sqrt', n_jobs=-1)
    model = dc.models.SklearnModel(sklearn_model)
    model.fit(train)

    return model


def _random_forest_model_builder(model_params, model_dir="."):
    """
    Search for the best parameters to build a random forest statistical
    model
    """
    sklearn_model = RandomForestRegressor(**model_params)
    return dc.models.SklearnModel(sklearn_model, model_dir)



