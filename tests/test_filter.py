"""Test the screening functionality."""

from .utils_test import PATH_TEST
from swan.filter.screen import validate_input


def test_filter_input() -> None:
    """Test that the input is handled correctly."""
    path_input_example = PATH_TEST / "input_test_filter.yml"
