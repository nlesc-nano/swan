"""Test the models funcionality."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from swan.input_validation import validate_input
from swan.models import FingerprintModeller, GraphModeller
from swan.models.modeller import main, predict_properties

path_input_test_fingerprints = Path("tests/test_files/input_test_fingerprint_train.yml")
path_input_test_graph = Path("tests/test_files/input_test_graph_train.yml")
path_trained_model = Path("tests/test_files/input_test_fingerprint_predict.yml")


def test_main(mocker):
    """Test the CLI for the models."""
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
        i=path_input_test_fingerprints, w=".", mode="train"))

    mocker.patch("swan.models.modeller.predict_properties", return_value=None)
    mocker.patch("swan.models.modeller.train_and_validate_model", return_value=None)
    main()


def test_split_data():
    """Check that training and validation set are independent."""
    opts = validate_input(path_input_test_fingerprints)
    researcher = FingerprintModeller(opts)
    researcher.split_data()
    xs = np.intersect1d(researcher.index_train, researcher.index_valid)
    assert xs.size == 0


def test_train_data_fingerprints(tmp_path):
    """Test that the dataset is trained properly."""
    opts = validate_input(path_input_test_fingerprints)

    # Use a temporal folde and train in CPU
    opts.model_path = os.path.join(tmp_path, "swan_models.pt")
    opts.use_cuda = False

    # Train for a few epoch
    opts.torch_config.epochs = 5
    opts.torch_config.batch_size = 500

    researcher = FingerprintModeller(opts)
    researcher.transform_labels()
    researcher.split_data()
    researcher.load_data()
    researcher.train_model()
    expected, predicted = researcher.evaluate_model()
    err = torch.functional.F.mse_loss(expected, predicted)
    assert os.path.exists(opts.model_path)
    assert err < 1.0


def test_predict_unknown_fingerprints():
    """Predict data for some smiles."""
    opts = validate_input(path_trained_model)
    opts.use_cuda = False
    df = predict_properties(opts)
    assert df['predicted_property'].notna().all()


def test_train_molecular_graph(tmp_path):
    """Test the training of convulution neural network on a molecular graph."""
    opts = validate_input(path_input_test_graph)

    # Use a temporal folde and train in CPU
    opts.model_path = os.path.join(tmp_path, "swan_models.pt")
    opts.use_cuda = False

    opts.torch_config.epochs = 5
    opts.torch_config.batch_size = 20

    researcher = GraphModeller(opts)
    researcher.transform_labels()
    researcher.split_data()
    researcher.load_data()
    researcher.train_model()
    expected, predicted = researcher.evaluate_model()
    assert os.path.exists(opts.model_path)
    err = torch.functional.F.mse_loss(expected, predicted)
    assert err.is_floating_point()

