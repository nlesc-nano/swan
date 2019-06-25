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


# Schemas to validate the input
sklearn_schema = Schema({
    # Use the SKlearn class
    "name": equal_lambda('sklearn'),
    # Use one of the following models
    "model": any_lambda(("randomforest", "svr")),

    # Input parameters for the model
    Optional("parameters", default={}): dict
})

schema_models = Schema({

    # Path to the csv file
    "csv_file": str,

    # Properties to predict
    "tasks": list,

    # Metric to evaluate the model
    Optional("metric", default='r2_score'): any_lambda(('r2_score')),

    # Method to get the features
    Optional("featurizer", default='circularfingerprint'):
    any_lambda(('circularfingerprint')),

    # What kind of methodology to use
    "interface": sklearn_schema,

    # Search for best hyperparameters
    Optional("optimize_hyperparameters", default=False): bool
})
