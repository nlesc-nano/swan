from schema import (And, Schema, SchemaError, Use)
from swan.utils import Options
import yaml


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


schema_models = Schema({

    # Path to the csv file
    "csv_file": str,

    # Properties to predict
    "tasks": list,

    # Method to get the features
    "featurizer": And(
        str, Use(str.lower), lambda s: s in (
            'circularfingerprint')
    )
})
