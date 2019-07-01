from pathlib import Path
from swan.models import (Modeler, ModelerSKlearn, ModelerTensorGraph)
from swan.models.models import main
from swan.models.input_validation import validate_input
from scipy.stats import linregress
import argparse

path_input_sklearn = Path("tests/test_files/input_test_sklearn.yml")
path_input_fcnet = Path("tests/test_files/input_test_fcnet.yml")


def test_main(mocker):
    """
    Test the CLI for the models
    """
    # Mock the CLI
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
        i=path_input_fcnet, w="."))

    # Mock the modelers
    mocker.patch("swan.models.models.ModelerSKlearn")
    mocker.patch("swan.models.models.ModelerTensorGraph")

    main()


def test_modeler_sklearn():
    """
    Check the instantiation of a ModelerSKlearn object
    """
    opts = validate_input(path_input_sklearn)
    researcher = Modeler(opts)

    xs = map(lambda x: getattr(researcher, x), ('metric', 'opts'))

    assert all((x is not None for x in xs))


def test_train_sklearn():
    """
    Check the training process of a sklearn model
    """
    opts = validate_input(path_input_sklearn)
    researcher = ModelerSKlearn(opts)

    model = researcher.train_model()

    rs = model.predict(researcher.data.test)

    data = researcher.data.test.y.reshape(rs.size)
    print("R2: ", linregress(rs, data))


def test_hyperparameters_sklearn():
    """
    Test the hyperparameters optimization
    """
    opts = validate_input(path_input_sklearn)
    opts.optimize_hyperparameters = True

    researcher = ModelerSKlearn(opts)
    researcher.train_model()


def test_train_tensorgraph():
    """
    Check the training process of a tensorgraph model
    """
    opts = validate_input(path_input_fcnet)
    researcher = ModelerTensorGraph(opts)

    model = researcher.train_model()

    rs = model.predict(researcher.data.test).flatten()

    data = researcher.data.test.y.reshape(rs.size)
    print("R2: ", linregress(rs, data))


def test_hyperparameters_tensorgraph():
    """
    Check the training process of a tensorgraph model
    """
    opts = validate_input(path_input_fcnet)
    researcher = ModelerTensorGraph(opts)
    opts.optimize_hyperparameters = True

    researcher.train_model()
