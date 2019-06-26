from pathlib import Path
from swan.models import Modeler
from swan.models.input_validation import validate_input

path_input = Path("tests/test_files/input_test_models.yml")


def test_input_validation():
    """
    Check that the input is validated correctly.
    """
    opts = validate_input(path_input)

    assert isinstance(opts, dict)


def test_modeler():
    """
    Check the instantiation of a Modeler object
    """
    opts = validate_input(path_input)
    researcher = Modeler(opts)

    xs = map(lambda x: getattr(researcher, x), ('metric', 'opts', 'available_models'))

    assert all((x is not None for x in xs))
