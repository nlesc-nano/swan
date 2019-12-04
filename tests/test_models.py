"""Test the models funcionality."""
import argparse
from pathlib import Path

import os
import numpy as np

from swan.models import Modeller
from swan.models.input_validation import validate_input
from swan.models.modeller import main, predict_properties

path_input_test = Path("tests/test_files/input_test_train.yml")
path_trained_model = Path("tests/test_files/input_test_predict.yml")


def test_main(mocker):
    """Test the CLI for the models."""
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
        i=path_input_test, w=".", mode="train"))

    mocker.patch("swan.models.modeller.predict_properties", return_value=None)
    mocker.patch("swan.models.modeller.train_and_validate_model", return_value=None)
    main()


def test_split_data():
    """Check that training and validation set are independent."""
    opts = validate_input(path_input_test)
    researcher = Modeller(opts)
    researcher.split_data()
    xs = np.intersect1d(researcher.index_train, researcher.index_valid)
    assert xs.size == 0


def test_train_data(tmp_path):
    """Test that the dataset is trained properly."""
    opts = validate_input(path_input_test)

    # Use a temporal folde and train in CPU
    opts.model_path = os.path.join(tmp_path, "swan_models.pt")
    opts.use_cuda = False

    # Train for a few epoch
    opts.torch_config.epochs = 5
    opts.torch_config.batch_size = 500

    researcher = Modeller(opts)
    researcher.transform_data()
    researcher.split_data()
    researcher.load_data()
    researcher.train_model()
    mean_loss = researcher.evaluate_model()
    assert os.path.exists(opts.model_path)
    assert mean_loss > 0 and mean_loss < 1e-1


def test_predict_unknown():
    """Predict data for some smiles."""
    opts = validate_input(path_trained_model)
    opts.use_cuda = False
    df = predict_properties(opts)
    assert df['predicted_property'].notna().all()


def test_save_dataset(tmp_path):
    """Test that the dataset is stored correctly."""
    return True
