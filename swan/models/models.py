# from .input_validation import validate_input
# from .plot import create_scatter_plot
# from pathlib import Path
# from swan.log_config import config_logger

# import argparse
import logging
# import numpy as np
# import pandas as pd

__all__ = ["Modeler"]

# Starting logger
logger = logging.getLogger(__name__)


# def main():
#     parser = argparse.ArgumentParser(description="train -i input.yml")
#     # configure logger
#     parser.add_argument('-i', required=True,
#                         help="Input file with options")
#     parser.add_argument("-m", "--mode", help="Operation mode: train or predict",
#                         choices=["train", "predict"], default="train")
#     parser.add_argument('-w', help="workdir", default=".")
#     args = parser.parse_args()

#     # start logger
#     config_logger(Path(args.w))

#     # log date
#     logger.info(f"Starting at:{datetime.now()}")

#     # Check that the input is correct
#     opts = validate_input(Path(args.i))
#     opts.mode = args.mode

#     if args.mode == "train":
#         train_model(opts)

#     else:
#         predict_properties(opts)


# def train_model(opts: dict) -> None:
#     """
#     Train the model usign the data specificied by the user
#     """
#     researcher = create_modeler(opts)
#     # train the model
#     if opts.load_model:
#         model = researcher.load_model()
#     else:
#         model = researcher.train_model()

#     # Check how good is the model
#     researcher.evaluate_model(model)

#     # # predict
#     rs = model.predict(researcher.data.test)

#     # Create a scatter plot of the predict values vs the ground true
#     create_scatter_plot(rs, researcher.data.test.y)


# def predict_properties(opts: dict) -> None:
#     """
#     Used a previous trained model to predict properties
#     """
#     def report(rs):
#         df = pd.read_csv(opts.dataset_file)
#         output = pd.DataFrame({'smiles': df['smiles'], 'predicted': rs})
#         output.to_csv("predicted.csv")

#     researcher = create_modeler(opts)
#     model = researcher.load_model()

#     # Prepare data
#     dataset = researcher.load_data()

#     # Predict
#     rs = model.predict(dataset).flatten()

#     if opts.report_predicted:
#         report(rs)

#     return rs


# def create_modeler(opts: dict):
#     """
#     Select the interface to use
#     """
#     # Train the model
#     return ModelerTensorGraph(opts)


class Modeler:

    def __init__(self, opts: dict):
        self.opts = opts

#         # Select featurizer and metric
#         self.select_featurizer()
#         self.select_metric()

#     def select_featurizer(self) -> None:
#         """
#         Use featurizer chosen by the user
#         """
#         logger.info(f"Using featurizer:{self.opts.featurizer}")
#         names = {
#             "circularfingerprint": "CircularFingerprint"
#         }
#         feat = import_module("deepchem.feat")
#         featurizer = getattr(feat, names[self.opts.featurizer])
#         self.featurizer = featurizer()

#     def select_metric(self) -> None:
#         """
#         Create instances of the metric to use
#         """
#         # Import the metric
#         mod_metric = import_module("deepchem.metrics")
#         try:
#             metric = getattr(mod_metric, self.opts.metric)
#             self.metric = dc.metrics.Metric(
#                 metric, np.mean, mode='regression')
#         except AttributeError:
#             print(f"Metric: {self.opts.metric} does not exist in deepchem")
#             raise

#     def load_data(self):
#         """
#         Load a dataset
#         """
#         logger.info(f"Loading data from {self.opts.dataset_file}")
#         print("file name: ", self.opts.dataset_file)
#         if Path(self.opts.dataset_file).suffix != ".csv":
#             dataset = dc.utils.save.load_from_disk(self.opts.dataset_file)

#         else:
#             tasks = [] if self.opts.mode == "predict" else self.opts.tasks
#             loader = dc.data.CSVLoader(tasks, smiles_field="smiles",
#                                        featurizer=self.featurizer)
#             dataset = loader.featurize(self.opts.dataset_file)

#         if self.opts.save_dataset:
#             file_name = Path(self.opts.workdir) / \
#                 f"{self.opts.filename_to_store_dataset}.joblib"
#             logger.info(f"saving dataset to: {file_name}")
#             dc.utils.save.save_to_disk(dataset, file_name)

#         return dataset

#     def load_model(self) -> Model:
#         """
#         Load model from disk
#         """
#         dataset = self.load_data()
#         self.split_data(dataset)

#         # Transform the y labels
#         if self.opts.mode == "train":
#             self.transform_data()

#         model = self.select_model()

#         return model.load_from_dir(self.opts.model_dir)

#     def split_data(self, dataset) -> None:
#         """
#         Split the entire dataset into a train, validate and test subsets.
#         """
#         logger.info("splitting the data into train, validate and test subsets")
#         splitter = dc.splits.ScaffoldSplitter()
#         self.data = DataSplitted(
#             *splitter.train_valid_test_split(dataset))

#     def transform_data(self):
#         """
#         Normalize the data to have zero-mean and unit-standard-deviation.
#         """
#         logger.info("Transforming the data")
#         self.transformers = [dc.trans.NormalizationTransformer(
#             transform_y=True, dataset=self.data.train)]
#         for ds in self.data:
#             for t in self.transformers:
#                 t.transform(ds)

#     def train_model(self) -> Model:
#         """
#         Use the data and `options` provided by the user to create an statistical
#         model.
#         """
#         dataset = self.load_data()

#         # Split the data into train/validation/test sets
#         self.split_data(dataset)

#         # Normalize the data
#         self.transform_data()

#         # Optimize hyperparameters
#         if self.opts["optimize_hyperparameters"]:
#             best_model, best_model_hyperparams, _ = self.optimize_hyperparameters()
#             logger.info(f"best hyperparameters: {best_model_hyperparams}")
#             return best_model
#         else:
#             # Use the statistical model as it is
#             return self.fit_model()

#     def evaluate_model(self, model) -> None:
#         """
#         Evaluate the predictive power of the model
#         """
#         evaluator = Evaluator(model, self.data.valid, self.transformers)
#         score = evaluator.compute_model_performance([self.metric])
#         logging.info(f"Score of the model is: {score}")
#         print("score: ", score)

#     def select_hyperparameters(self) -> dict:
#         """
#         Use the parameters provided by the user or the defaults
#         """
#         model_name = self.opts.interface["model"]
#         if not self.opts.interface["parameters"]:
#             return default_hyperparameters.get(model_name, {})
#         else:
#             return self.opts.interface["parameters"]

#     def optimize_hyperparameters(self) -> tuple:
#         """
#         Search for the best hyperparameters for a given model
#         """
#         model_name = self.opts.interface["model"]

#         model_class = partial(_model_builder_tensorgraph,
#                               self.n_tasks, self.n_features)

#         optimizer = dc.hyper.HyperparamOpt(model_class)
#         params_dict = data_hyperparam_search[model_name]
#         best_model, best_model_hyperparams, all_models_results = optimizer.hyperparam_search(
#             params_dict, self.data.train, self.data.valid, self.transformers,
#             metric=self.metric)

#         return best_model, best_model_hyperparams, all_models_results


# class ModelerTensorGraph(Modeler):

#     def __init__(self, opts: dict):
#         super().__init__(opts)
#         self.available_models = {
#             'multitaskregressor': MultitaskRegressor,
#         }
#         self.n_tasks = len(self.opts.tasks)
#         self.n_features = getattr(self.featurizer, 'size', None)

#     def select_model(self) -> Model:
#         """
#         Return the Model object specificied by the user
#         """
#         model_name = self.opts.interface["model"]
#         logger.info(f"Train the model using {model_name}")

#         # Select model and fit it
#         hyper = self.select_hyperparameters()

#         # Operation mode
#         hyper["mode"] = 'regression'

#         # Path to store the trained model
#         hyper["model_dir"] = self.opts.model_dir

#         logger.info(f"Model hyperparameters are: {hyper}")

#         # Create model
#         tensorgraph_model = self.available_models[model_name]
#         args = self._get_positional_args()
#         return tensorgraph_model(*args, **hyper)

#     def _get_positional_args(self) -> list:
#         """
#         Select the positional arguments for a given model
#         """
#         model_name = self.opts.interface["model"]
#         if model_name == 'multitaskregressor':
#             return [self.n_tasks, self.n_features]
#         else:
#             return [self.n_tasks]

#     def fit_model(self) -> Model:
#         """
#         Fit the statistical model using the given hyperparameters or the default
#         """
#         model = self.select_model()
#         model.fit(self.data.train, nb_epoch=self.opts.interface["epochs"])

#         # save model data
#         model.save()

#         return model


# def _model_builder_tensorgraph(n_tasks: int, n_features: int, model_params: dict,
#                                model_dir: str = "."):
#     """
#     Create a TensorGraph Model
#     """
#     return MultitaskRegressor(n_tasks, n_features, **model_params)
