from pathlib import Path
from schema import SchemaError
from swan.models.input_validation import (
    SCHEMA_TORCH, validate_input)
import yaml


path_input_train = Path("tests/test_files/input_test_train.yml")


def test_input_validation():
    """Check that the input is validated correctly."""
    opts = validate_input(path_input_train)

    assert isinstance(opts, dict)


def test_wrong_input(tmp_path):
    """Test schema failure with wrong input."""
    d = {"csv_file": "Non-existing/path.csv"}

    file_path = Path(tmp_path) / "tmp.yml"

    with open(file_path, "w") as f:
        yaml.dump(d, f)

    try:
        validate_input(file_path)
    except SchemaError:
        pass


def test_schema_torch():
    """Check input for tensorgraph models."""
    d = {"epochs": 100, "batch_size": 100}
    SCHEMA_TORCH.validate(d)
