"""Test the screening functionality."""

from pathlib import Path
from .utils_test import PATH_TEST
from swan.filter.screen import main, apply_filters, read_molecules
from swan.utils import Options
from typing import Mapping, TypeVar
import argparse
import pandas as pd
import os

T = TypeVar("T")

path_input_test_filter = PATH_TEST / "input_test_filter.yml"


def run_workflow(opts: Options) -> pd.DataFrame:
    """Apply the filters and read the output."""
    apply_filters(opts)
    filter_mols = read_molecules(opts.output_file)
    return filter_mols


def create_options(filters: Mapping[str, T], smiles_file: str, tmp_path: str) -> Options:
    """Create Options object to filter."""
    opts = Options()
    opts.smiles_file = (PATH_TEST / smiles_file).absolute().as_posix()
    opts.filters = filters
    opts.output_file = f"{tmp_path}/candidates.csv"
    opts.workdir = tmp_path

    return opts


def remove_output(output_file: str) -> None:
    """Remove the output file if exists."""
    path = Path(output_file)
    if path.exists():
        os.remove(path)


def test_filter_cli(mocker) -> None:
    """Test that the CLI works correctly."""
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
        i=path_input_test_filter))

    mocker.patch("swan.filter.screen.apply_filters", return_value=None)
    main()


def test_filter_functional_groups(tmp_path) -> None:
    """Test that the functional group filter is applied properly."""
    smiles_file = "smiles_functional_groups.csv"
    filters = {"functional_groups": ["C(=O)O"]}
    opts = create_options(filters, smiles_file, tmp_path)
    try:
        filter_mols = run_workflow(opts)
        expected = ("O=C(O)C1CNC2C3CC4C2N4C13", "C#CC12CC(CO1)NCC2C(=O)O",
                    "CCCCCCCCC=CCCCCCCCC(=O)O", "CC(=O)O",
                    "O=C(O)Cc1ccccc1", "CC(O)C(=O)O")

        print(f"candidates:\n{filter_mols}")
        assert all(mol in filter_mols.smiles.values for mol in expected)
        assert len(filter_mols.smiles) == 6

    finally:
        remove_output(opts.output_file)


def test_filter_bulkiness(tmp_path) -> None:
    """Test that the bulkiness filter is applied properly."""
    smiles_file = "smiles_carboxylic.csv"
    filters = {"bulkiness": {"lower_than": 20}}
    opts = create_options(filters, smiles_file, tmp_path)
    opts.core = PATH_TEST / "Cd68Se55.xyz"
    opts.anchor = "O(C=O)[H]"
    try:
        filter_mols = run_workflow(opts)
    finally:
        remove_output(opts.output_file)
