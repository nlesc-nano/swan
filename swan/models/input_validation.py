from schema import (And, Optional, Schema, SchemaError, Use)
from swan.utils import Options
import yaml


def equal_lambda(name: str):
    """
    Create an schema checking that the keyword matches the expected value
    """
    return And(
        str, Use(str.lower), lambda s: s == name)


def any_lambda(xs: iter):
    """
    Create an schema checking that the keyword matches one of the expected values
    """
    return And(
        str, Use(str.lower), lambda s: s in xs)


def validate_input(file_input: str) -> Options:
    """
    Check the input validation against an schema
    """
    with open(file_input, 'r') as f:
        dict_input = yaml.load(f.read(), Loader=yaml.FullLoader)
    try:
        d = schema_modeler.validate(dict_input)
        return Options(d)

    except SchemaError as e:
        msg = "There was an error in the input yaml provided:\n{}".format(e)
        print(msg)
        raise


schema_torch = Schema({
    # Number of epoch to train for
    Optional("epochs", default=100): int,

    # Method to get the features
    Optional("featurizer", default='circularfingerprint'): any_lambda(('circularfingerprint')),

    # Metric to evaluate the model
    Optional("metric", default='r2_score'): str,

    # Frequency to log the ressult between epochs
    Optional("frequency_log_epochs", default=10): int
})

schema_modeler = Schema({
    # Load the dataset from a file
    "dataset_file": str,

    # Network and training options options
    Optional("torch_config"): schema_torch,

    # Search for best hyperparameters
    Optional("optimize_hyperparameters", default=False): bool,

    # Save the dataset to a file
    Optional("save_dataset", default=True): bool,

    # Load model from disk
    Optional("load_model", default=False): bool,

    # Report predicted data
    Optional('report_predicted', default=True): bool,

    # Folder to save the models
    Optional("model_dir", default="swan_models"): str,

    Optional("filename_to_store_dataset", default="dataset"): str,

    # Workdir
    Optional("workdir", default="."): str
})
