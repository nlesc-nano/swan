"""Functionality to validate the user input against the schemas."""

import warnings

import torch
import yaml
from schema import And, Optional, Schema, SchemaError, Use

from swan.utils import Options


def equal_lambda(name: str):
    """Create an schema checking that the keyword matches the expected value."""
    return And(
        str, Use(str.lower), lambda s: s == name)


def any_lambda(array: iter):
    """Create an schema checking that the keyword matches one of the expected values."""
    return And(
        str, Use(str.lower), lambda s: s in array)


def validate_input(file_input: str) -> Options:
    """Check the input validation against an schema."""
    with open(file_input, 'r') as f:
        dict_input = yaml.load(f.read(), Loader=yaml.FullLoader)
    try:
        data = SCHEMA_MODELER.validate(dict_input)
        opts = Options(data)
        if opts.use_cuda:
            check_if_cuda_is_available(opts)
        return opts

    except SchemaError as err:
        msg = "There was an error in the input yaml provided:\n{}".format(err)
        print(msg)
        raise


def check_if_cuda_is_available(opts: dict):
    """Check that a CUDA device is available, otherwise turnoff the option."""
    if not torch.cuda.is_available():
        opts.use_cuda = False
        warnings.warn("There is not CUDA device available using default CPU methods")


SCHEMA_OPTIMIZER = Schema({
    # Learning rate
    Optional("lr", default=0.1): float,

    Optional("momentum", default=0): float,

    Optional("dampening", default=0): float,

    Optional("weight_decay", default=0): float
})


OPTIMIZER_DEFAULTS = SCHEMA_OPTIMIZER.validate({})

SCHEMA_TORCH = Schema({

    # Number of epoch to train for
    Optional("epochs", default=100): int,

    Optional("batch_size", default=100): int,

    Optional("optimizer", default=OPTIMIZER_DEFAULTS): SCHEMA_OPTIMIZER,

    # Method to get the features
    Optional("featurizer", default='circularfingerprint'): any_lambda(('circularfingerprint')),

    # Metric to evaluate the model
    Optional("metric", default='r2_score'): str,

    # Frequency to log the ressult between epochs
    Optional("frequency_log_epochs", default=10): int,

})

SCHEMA_MODELER = Schema({
    # Load the dataset from a file
    "dataset_file": str,

    # Property to predict
    "property": str,

    # Whether to use CPU or GPU
    Optional("use_cuda", default=False): bool,

    # Network and training options options
    Optional("torch_config"): SCHEMA_TORCH,

    # Search for best hyperparameters
    Optional("optimize_hyperparameters", default=False): bool,

    # Save the dataset to a file
    Optional("save_dataset", default=True): bool,

    # Folder to save the models
    Optional("model_path", default="swan_models.pt"): str,

    # Report predicted data
    Optional('report_predicted', default=True): bool,

    # Workdir
    Optional("workdir", default="."): str
})
