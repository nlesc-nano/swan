"""Check the input schemas."""

from pathlib import Path

import pytest
import yaml
from schema import SchemaError

from swan.utils.input_validation import SCHEMA_TORCH, validate_input
from .utils_test import PATH_TEST

path_input_train = PATH_TEST / "input_test_fingerprint_train.yml"


def test_input_validation():
    """Check that the input is validated correctly."""
    opts = validate_input(path_input_train)

    assert isinstance(opts, dict)


def test_wrong_input(tmp_path: Path):
    """Test schema failure with wrong input."""
    d = {"csv_file": "Non-existing/path.csv"}

    file_path = Path(tmp_path) / "tmp.yml"

    with open(file_path, "w") as f:
        yaml.dump(d, f)

    with pytest.raises(SchemaError) as excinfo:
        validate_input(file_path)

    assert "Missing keys" in str(excinfo.value)


def test_schema_torch():
    """Check input for tensorgraph models."""
    d = {"epochs": 100, "batch_size": 100}
    SCHEMA_TORCH.validate(d)
