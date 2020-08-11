"""Functionality to validate the user input against the schemas."""

import warnings

import torch
import yaml
from schema import And, Optional, Or, Schema, SchemaError, Use

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
        msg = f"There was an error in the input yaml provided:\n{err}"
        print(msg)
        raise


def check_if_cuda_is_available(opts: dict):
    """Check that a CUDA device is available, otherwise turnoff the option."""
    if not torch.cuda.is_available():
        opts.use_cuda = False
        warnings.warn("There is not CUDA device available using default CPU methods")


SCHEMA_OPTIMIZER = Schema({

    Optional("name", default="sgd"): any_lambda(("adam", "sgd")),

    # Learning rate
    Optional("lr", default=0.1): float,

    # Momentum
    Optional("momentum", default=0): float,

    # Nesterov accelerated gradient
    Optional("nesterov", default=False): bool

})


OPTIMIZER_DEFAULTS = SCHEMA_OPTIMIZER.validate({})

SCHEMA_TORCH = Schema({

    # Number of epoch to train for
    Optional("epochs", default=100): int,

    Optional("batch_size", default=100): int,

    Optional("loss_function", default="MSELoss"): str,

    Optional("optimizer", default=OPTIMIZER_DEFAULTS): SCHEMA_OPTIMIZER
})

TORCH_DEFAULTS = SCHEMA_TORCH.validate({})


SCHEMA_MODEL_FINGERPRINTS = Schema({
    Optional("input_cells", default=2048): int,

    Optional("hidden_cells", default=1000): int
})
MODEL_DEFAULTS = SCHEMA_MODEL_FINGERPRINTS.validate({})

SCHEMA_FINGERPRINTS = Schema({
    Optional("fingerprint", default='atompair'): any_lambda(('morgan', 'atompair', 'torsion'))
})

SCHEMA_GRAPH = Schema({
    "molecular_graph": dict
})

SCHEMA_MODELER = Schema({
    # Load the dataset from a file
    "dataset_file": str,

    # Property to predict
    "property": str,

    # Method to get the features
    "featurizer": Or(SCHEMA_FINGERPRINTS, equal_lambda("molecular_graph")),

    # Whether to use CPU or GPU
    Optional("use_cuda", default=False): bool,

    Optional("model", default=MODEL_DEFAULTS): Or(SCHEMA_MODEL_FINGERPRINTS),

    # Network and training options options
    Optional("torch_config", default=TORCH_DEFAULTS): SCHEMA_TORCH,

    # Folder to save the models
    Optional("model_path", default="swan_models.pt"): str,

    # Workdir
    Optional("workdir", default="."): str
})
