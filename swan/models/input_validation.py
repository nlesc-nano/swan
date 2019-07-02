from schema import (And, Optional, Or, Schema, SchemaError, Use)
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


def validate_input(file_input: str):
    """
    Check the input validation against an schema
    """
    with open(file_input, 'r') as f:
        dict_input = yaml.load(f.read(), Loader=yaml.FullLoader)
    try:
        d = schema_models.validate(dict_input)
        return Options(d)

    except SchemaError as e:
        msg = "There was an error in the input yaml provided:\n{}".format(e)
        print(msg)
        raise


# Schemas to validate the input
sklearn_schema = Schema({
    # Use the SKlearn class
    "name": equal_lambda('sklearn'),
    # Use one of the following models
    "model": any_lambda(("randomforest", "svr", "kernelridge", "bagging")),

    # Input parameters for the model
    Optional("parameters", default={}): dict
})

tensorgraph_schema = Schema({
    # Use the tensorgraph class
    "name": equal_lambda('tensorgraph'),

    # Available models
    "model": str,

    # Number of epoch to train for
    Optional("epochs", default=10): int,

    # Input parameters for the model
    Optional("parameters", default={}): dict
})

schema_models = Schema({

    # Load the dataset from a file
    "dataset_file": str,

    # Properties to predict
    "tasks": list,

    # Metric to evaluate the model
    Optional("metric", default='r2_score'): any_lambda(('r2_score')),

    # Method to get the features
    Optional("featurizer", default='circularfingerprint'):
    any_lambda(('circularfingerprint')),

    # What kind of methodology to use
    "interface": Or(sklearn_schema, tensorgraph_schema),

    # Search for best hyperparameters
    Optional("optimize_hyperparameters", default=False): bool,

    # Save the dataset to a file
    Optional("save_dataset", default=True): bool,

    # Load model from disk
    Optional("load_model", default=False): bool,

    # Folder to save the models
    Optional("model_dir", default="models"): str,

    Optional("filename_to_store_dataset", default="dataset"): str,

    # Workdir
    Optional("workdir", default="."): str
})
