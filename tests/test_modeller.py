"""Test the models funcionality."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from pytest_mock import MockFixture

from swan.utils.input_validation import validate_input
from swan.modeller import (FingerprintModeller, GraphModeller)

from .utils_test import PATH_TEST

PATH_INPUT_TEST_FINGERPRINTS = PATH_TEST / "input_test_fingerprint_train.yml"

# def test_main(mocker: MockFixture):
#     """Test the CLI for the models."""
#     mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
#         i=PATH_INPUT_TEST_FINGERPRINTS, w=".", mode="train", restart=False))

#     mocker.patch("swan.modeller.predict_properties", return_value=None)
#     main()

#     mocker.patch("swan.modeller.train_and_validate_model", return_value=None)


def test_split_data():
    """Check that training and validation set are independent."""
    opts = validate_input(PATH_INPUT_TEST_FINGERPRINTS)
    researcher = FingerprintModeller(opts)
    researcher.split_data()
    xs = np.intersect1d(researcher.index_train, researcher.index_valid)
    assert xs.size == 0


def test_train_data_fingerprints(tmp_path: Path):
    """Test that the dataset is trained properly."""
    opts = validate_input(PATH_INPUT_TEST_FINGERPRINTS)

    # Use a temporal folde and train in CPU
    opts.model_path = os.path.join(tmp_path, "swan_models.pt")
    opts.use_cuda = False

    # Train for a few epoch
    opts.torch_config.epochs = 5
    opts.torch_config.batch_size = 500

    researcher = FingerprintModeller(opts)
    researcher.scale_labels()
    researcher.split_data()
    researcher.load_data()
    researcher.train_model()
    expected, predicted = researcher.validate_model()
    err = torch.functional.F.mse_loss(expected, predicted)
    assert os.path.exists(opts.model_path)
    assert not np.isnan(err.item())


def test_predict_unknown_fingerprints():
    """Predict data for some smiles."""
    opts = validate_input(PATH_TEST / "input_test_fingerprint_predict.yml")
    opts.use_cuda = False
    opts.mode = "predict"
    df = predict_properties(opts)
    assert df['predicted_property'].notna().all()


def test_train_molecular_graph(tmp_path: Path):
    """Test the training of convulution neural network on a molecular graph."""
    opts = validate_input(PATH_TEST / "input_test_graph_train.yml")

    # Use a temporal folde and train in CPU
    opts.model_path = os.path.join(tmp_path, "swan_models.pt")
    opts.use_cuda = False

    opts.torch_config.epochs = 5
    opts.torch_config.batch_size = 20

    researcher = GraphModeller(opts)
    researcher.scale_labels()
    researcher.split_data()
    researcher.load_data()
    researcher.train_model()
    expected, predicted = researcher.validate_model()
    assert os.path.exists(opts.model_path)
    err = torch.functional.F.mse_loss(expected, predicted)
    assert not np.isnan(err.item())


def test_predict_unknown_graph():
    """Predict properties using the graph model."""
    opts = validate_input(PATH_TEST / "input_test_graph_predict.yml")
    opts.use_cuda = False
    opts.mode = "predict"
    df = predict_properties(opts)
    assert df['predicted_property'].notna().all()


def test_load_geometries(tmp_path: Path):
    """Check that the geometries are load correctly."""
    opts = validate_input(PATH_TEST / "input_test_graph_geometries.yml")

    # Use a temporal folde and train in CPU
    opts.model_path = os.path.join(tmp_path, "swan_models.pt")
    opts.model_scales = os.path.join(tmp_path, "model_scales.pkl")
    opts.use_cuda = False

    opts.torch_config.epochs = 5
    opts.torch_config.batch_size = 20

    researcher = GraphModeller(opts)
    researcher.scale_labels()
    researcher.split_data()
    researcher.load_data()
