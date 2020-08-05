"""Test the screening functionality."""

from pathlib import Path
from .utils_test import PATH_TEST
from swan.filter.screen import main, apply_filters, read_molecules
from swan.utils import Options
import argparse
import os

path_input_test_filter = PATH_TEST / "input_test_filter.yml"


def test_filter_cli(mocker) -> None:
    """Test that the CLI works correctly."""
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
        i=path_input_test_filter))

    mocker.patch("swan.filter.screen.apply_filters", return_value=None)
    main()


def test_filter_functional_groups(tmp_path) -> None:
    """Test that the filters are applied properly."""
    opts = Options()
    opts.smiles_file = "tests/test_files/functional_groups.csv"
    opts.filters = {"functional_groups": ["C(=O)O"]}
    opts.output_file = f"{tmp_path}/candidates.csv"
    try:
        apply_filters(opts)
        filter_mols = read_molecules(opts.output_file)
        smiles = filter_mols["smiles"]
        expected = ("OC(=O)C1CNC2C3C4CC2C1N34", "OC(=O)C1CNC2COC1(C2)C#C")

        print(f"candidates:\n{filter_mols}")
        assert all(mol in smiles.values for mol in expected)
        assert len(smiles) == 2

    finally:
        path = Path(opts.output_file)
        if path.exists():
            os.remove(path)
