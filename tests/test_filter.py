"""Test the screening functionality."""

import argparse
import shutil
from pathlib import Path
from typing import List, Mapping, TypeVar

import pandas as pd

from swan.filter.screen import main, split_filter_in_batches
from swan.utils import Options, read_molecules

from .utils_test import PATH_TEST

T = TypeVar("T")

PATH_INPUT_TEST_FILTER = PATH_TEST / "input_test_filter.yml"


def run_workflow(opts: Options) -> pd.DataFrame:
    """Apply the filters and read the output."""
    split_filter_in_batches(opts)
    filter_mols = read_molecules("results/batch_0/candidates.csv")
    return filter_mols


def create_options(filters: Mapping[str, T], smiles_file: str, tmp_path: str) -> Options:
    """Create Options object to filter."""
    opts = Options()
    opts.smiles_file = (PATH_TEST / smiles_file).absolute().as_posix()
    opts.filters = filters
    opts.output_path = "results"
    opts.workdir = tmp_path
    opts.batch_size = 100

    return opts


def remove_output(output_path: str) -> None:
    """Remove the output file if exists."""
    path = Path(output_path)
    if path.exists():
        shutil.rmtree(path)


def check_expected(opts: Options, expected: List[str]) -> None:
    """Run a filter workflow using `opts` and check the results."""
    try:
        computed = run_workflow(opts)
        print("expected:\n", expected)
        print(f"candidates:\n{computed.smiles.values}")
        assert all(mol in computed.smiles.values for mol in expected)
        assert len(computed.smiles) == len(expected)

    finally:
        remove_output(opts.output_path)


def test_filter_cli(mocker) -> None:
    """Test that the CLI works correctly."""
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
        i=PATH_INPUT_TEST_FILTER))

    mocker.patch("swan.filter.screen.split_filter_in_batches", return_value=None)
    main()


def test_contain_functional_groups(tmp_path) -> None:
    """Test that the functional group filter is applied properly."""
    smiles_file = "smiles_functional_groups.csv"
    filters = {"include_functional_groups": ["C(=O)O"]}
    opts = create_options(filters, smiles_file, tmp_path)
    expected = ("O=C(O)C1CNC2C3CC4C2N4C13", "C#CC12CC(CO1)NCC2C(=O)O",
                "CCCCCCCCC=CCCCCCCCC(=O)O", "CC(=O)O",
                "O=C(O)Cc1ccccc1", "CC(O)C(=O)O")
    check_expected(opts, expected)


def test_exclude_functional_groups(tmp_path) -> None:
    """Test that some functional groups are excluded correctly."""

    smiles_file = "smiles_functional_groups.csv"
    filters = {"exclude_functional_groups": ["CN", "C#C"]}
    opts = create_options(filters, smiles_file, tmp_path)
    expected = ("c1ccccc1", "CCO", "CCCCCCCCC=CCCCCCCCC(=O)O",
                "CC(=O)O", "O=C(O)Cc1ccccc1", "CC(O)C(=O)O")
    check_expected(opts, expected)


def test_filter_bulkiness(tmp_path) -> None:
    """Test that the bulkiness filter is applied properly."""
    smiles_file = "smiles_carboxylic.csv"
    filters = {"bulkiness": {"lower_than": 20}}
    opts = create_options(filters, smiles_file, tmp_path)
    opts.core = PATH_TEST / "Cd68Se55.xyz"
    opts.anchor = "O(C=O)[H]"

    expected = ("CCCCCCCCC=CCCCCCCCC(=O)O", "CC(=O)O", "CC(O)C(=O)O")
    check_expected(opts, expected)


def test_filter_scscore(tmp_path) -> None:
    """Test that the scscore filter is applied properly."""
    smiles_file = "smiles_carboxylic.csv"
    filters = {"scscore": {"lower_than": 1.3}}
    opts = create_options(filters, smiles_file, tmp_path)

    expected = ("CC(=O)O",)
    check_expected(opts, expected)
