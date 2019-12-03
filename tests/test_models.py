"""Test the models funcionality."""
from swan.models.models import main
from pathlib import Path
import argparse

path_input_test = Path("tests/test_files/input_test_train.yml")


def test_main(mocker):
    """Test the CLI for the models."""
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
        i=path_input_test, w=".", mode="train"))

    mocker.patch("swan.models.models.predict_properties", return_value=None)
    mocker.patch("swan.models.models.train_and_validate_model", return_value=None)

    main()


def test_load_data():
    """Test that the data is loaded correctly."""
    return True


def test_load_model():
    """
    Test that a model is loaded correctly
    """
    return True


def test_predict_unknown():
    """
    Predict data for a some smiles
    """
    return True


def test_save_dataset(tmp_path):
    """
    Test that the dataset is stored correctly
    """
    return True


def check_predict(model, researcher) -> bool:
    """Check that the predicted numbers are real. """
    return True
