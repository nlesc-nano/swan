from pathlib import Path
from schema import SchemaError
from swan.models.input_validation import (
    sklearn_schema, tensorgraph_schema, validate_input)
import yaml


path_input_sklearn = Path("tests/test_files/input_test_sklearn.yml")


def test_input_validation():
    """
    Check that the input is validated correctly.
    """
    opts = validate_input(path_input_sklearn)

    assert isinstance(opts, dict)


def test_wrong_input(tmp_path):
    """
    Test schema failure with wrong input
    """
    d = {
        "csv_file": "Non-existing/path.csv",
        "tasks": ['one'],
        "interface": {'name': "SKlearn", "model": "randomForest", "parameters": {"n_jobs": -1}}
    }

    file_path = Path(tmp_path) / "tmp.yml"

    with open(file_path, "w") as f:
        yaml.dump(d, f)

    try:
        validate_input(file_path)
    except SchemaError:
        pass


def test_sklearn_schema():
    """
    Check input for sklearn models
    """
    d = {'name': "SKlearn", "model": "randomForest", "parameters": {"n_jobs": -1}}
    sklearn_schema.validate(d)


def test_tensorgraph_schema():
    """
    Check input for tensorgraph models
    """
    d = {'name': "TensorGraph", "model": "fcnet",
         "epochs": 100, "parameters": {"dropout": 0.75}}
    tensorgraph_schema.validate(d)
