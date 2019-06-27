from pathlib import Path
from swan.models import (ModelerSKlearn, ModelerTensorGraph)
from swan.models.input_validation import validate_input
from scipy.stats import linregress
import pytest

path_input = Path("tests/test_files/input_test_models.yml")


def test_input_validation():
    """
    Check that the input is validated correctly.
    """
    opts = validate_input(path_input)

    assert isinstance(opts, dict)


def test_modeler_sklearn():
    """
    Check the instantiation of a ModelerSKlearn object
    """
    opts = validate_input(path_input)
    researcher = ModelerSKlearn(opts)

    xs = map(lambda x: getattr(researcher, x), ('metric', 'opts', 'available_models'))

    assert all((x is not None for x in xs))


def test_train_sklearn():
    """
    Check the training process of a sklearn model
    """
    opts = validate_input(path_input)
    researcher = ModelerSKlearn(opts)

    model = researcher.train_model()

    rs = model.predict(researcher.data.test)

    data = researcher.data.test.y.reshape(rs.size)
    print("R2: ", linregress(rs, data))


def test_modeler_tensorgraph():
    """
    Check the instantiation of a ModelerTensorGraph object
    """
    pass


@pytest.mark.xfail
def test_train_tensorgraph():
    """
    Check the training process of a tensorgraph model
    """
    opts = validate_input(path_input)
    researcher = ModelerTensorGraph(opts)

    model = researcher.train_model()

    rs = model.predict(researcher.data.test)

    data = researcher.data.test.y.reshape(rs.size)
    print("R2: ", linregress(rs, data))
