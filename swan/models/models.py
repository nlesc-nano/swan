from .input_validation import validate_input
from collections import namedtuple
from deepchem.utils.evaluate import Evaluator
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from swan.log_config import config_logger

import argparse
import logging
import pandas as pd
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
    researcher.train_model()


class Modeler:

    def __init__(self, opts: dict):
        self.opts = opts

    def train_model(self):
        """
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
        model = self.call_random_forest()

        # Check how good is the model
        self.evaluate_model(model)

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

    def call_random_forest(self):
        """
        Call the sklearn `RandomForestRegressor`
        """
        logger.info("Train the model using a random forest")
        sklearn_model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        model = dc.models.SklearnModel(sklearn_model)
        model.fit(self.data.train)

        return model

    def evaluate_model(self, model):
        """
        Evaluate the predictive power of the model
        """
        metric = dc.metrics.Metric(dc.metrics.r2_score)
        evaluator = Evaluator(model, self.data.valid, self.transformers)
        score = evaluator.compute_model_performance([metric])
        print("score: ", score)
