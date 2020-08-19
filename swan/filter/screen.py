""""
Module to screen smile by functional group and other properties.

Index
-----
.. currentmodule:: swan.filter.screen
.. autosummary::
    {autosummary}

API
---
{autodata}

"""

import argparse
import logging
import pkg_resources
import sys
import tempfile
from functools import partial
from numbers import Real
from pathlib import Path
from typing import FrozenSet

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from schema import Optional, Or, Schema, SchemaError

from ..cosmo.cat_interface import call_cat_in_parallel
from ..log_config import configure_logger
from ..utils import Options

logger = logging.getLogger(__name__)

#: Schema to validate the ordering keywords
SCHEMA_ORDERING = Or(
    Schema({"greater_than": Real}),
    Schema({"lower_than": Real}))

#: Schema to validate the filters to apply for screening
SCHEMA_FILTERS = Schema({
    # Include or exclude one or more functional group using smiles
    Optional("include_functional_groups"): Schema([str]),
    Optional("exclude_functional_groups"): Schema([str]),

    # Select smiles >, < or = to some value
    Optional("bulkiness"): SCHEMA_ORDERING,

    Optional("SA_score"): SCHEMA_ORDERING
})

#: Schema to validate the input for screening
SCHEMA_SCREEN = Schema({
    # Load the dataset from a file
    "smiles_file": str,

    # Constrains to filter
    "filters": SCHEMA_FILTERS,

    # Functional group used as anchor
    Optional("anchor", default="O(C=O)[H]"): str,

    # path to the molecular coordinates of the Core to attach the ligands
    Optional("core"): str,

    # path to the workdir
    Optional("workdir", default=tempfile.mkdtemp(prefix="swan_workdir_")): str,

    # Number of molecules to compute simultaneously
    Optional("batch_size", default=1000): int,

    # File to print the final candidates
    Optional("output_path", default="results"): str
})


def read_molecules(input_file: Path) -> pd.DataFrame:
    """Read smiles from a csv-like file."""
    df = pd.read_csv(input_file).reset_index(drop=True)
    # remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def validate_input(file_input: str) -> Options:
    """Check the input validation against an schema."""
    with open(file_input, 'r') as f:
        dict_input = yaml.load(f.read(), Loader=yaml.FullLoader)
    try:
        data = SCHEMA_SCREEN.validate(dict_input)
        return Options(data)
    except SchemaError as err:
        msg = f"There was an error in the input yaml provided:\n{err}"
        print(msg)
        raise


def split_filter_in_batches(opts: Options) -> None:
    """Split the computations into smaller batches that fit into memory."""
    # Read molecules into a pandas dataframe
    molecules = read_molecules(opts.smiles_file)

    # Create folder to store the output
    result_path = Path(opts.output_path)
    result_path.mkdir(exist_ok=True, parents=True)

    # Compute the number of batches and split
    nsmiles = len(molecules)
    number_of_batches = nsmiles // opts.batch_size
    number_of_batches = number_of_batches if number_of_batches > 0 else 1

    for k, batch in enumerate(np.array_split(molecules, number_of_batches)):
        logger.info(f"computing batch: {k}")
        output_file = create_ouput_file(result_path, k)
        try:
            apply_filters(batch, opts, output_file)
        except:
            error = next(iter(sys.exc_info()))
            logger.error(error)


def apply_filters(molecules: pd.DataFrame, opts: Options, output_file: Path) -> None:
    """Apply a set of filters to the given smiles."""
    logger.info("converting smiles to rdkit molecules")
    # Create rdkit representations
    converter = np.vectorize(Chem.MolFromSmiles)
    molecules["rdkit_molecules"] = converter(molecules.smiles)

    # Remove invalid molecules
    molecules = molecules[molecules.rdkit_molecules.notnull()]

    # Convert smiles to the standard representation
    back_converter = np.vectorize(Chem.MolToSmiles)
    molecules.smiles = back_converter(molecules.rdkit_molecules)

    # Apply all the filters
    available_filters = {
        "include_functional_groups": include_functional_groups,
        "exclude_functional_groups": exclude_functional_groups,
        "bulkiness": filter_by_bulkiness}

    for key in opts.filters.keys():
        if key in available_filters:
            molecules = available_filters[key](molecules, opts)

    molecules.to_csv(output_file, columns=["smiles"])
    logger.info(f"The filtered candidates has been written to the {output_file} file!")


def include_functional_groups(molecules: pd.DataFrame, opts: Options) -> pd.DataFrame:
    """Check that the molecules contain some functional groups."""
    groups = opts["filters"]["include_functional_groups"]
    logger.info(f"including molecules that contains the groups: {groups}")
    return filter_by_functional_group(molecules, opts, "include_functional_groups", False)


def exclude_functional_groups(molecules: pd.DataFrame, opts: Options) -> pd.DataFrame:
    """Check that the molecules do not contain some functional groups."""
    groups = opts["filters"]["exclude_functional_groups"]
    logger.info(f"exclude molecules that contains the groups: {groups}")
    return filter_by_functional_group(molecules, opts, "exclude_functional_groups", True)


def filter_by_functional_group(molecules: pd.DataFrame, opts: Options, key: str,
                               exclude: bool) -> pd.DataFrame:
    """Search for a set of functional_groups."""
    # Transform functional_groups to rkdit molecules
    functional_groups = opts["filters"][key]
    patterns = {Chem.MolFromSmiles(f) for f in functional_groups}

    # Function to apply predicate
    pattern_check = np.vectorize(partial(has_substructure, exclude, patterns))

    # Check if the functional_groups are in the molecules
    has_pattern = pattern_check(molecules["rdkit_molecules"])

    return molecules[has_pattern]


def has_substructure(exclude: bool, patterns: FrozenSet, mol: Chem.Mol) -> bool:
    """Check if there is any element of `pattern` in `mol`."""
    result = False if mol is None else any(mol.HasSubstructMatch(p) for p in patterns)
    if exclude:
        return not result
    return result


def filter_by_bulkiness(molecules: pd.DataFrame, opts: Options) -> pd.DataFrame:
    """Filter the ligands that have a given bulkiness.

    The bulkiness is computed using the CAT library: https://github.com/nlesc-nano/CAT
    The user must specify whether the bulkiness should be lower_than, greater_than
    or equal than a given value.
    """
    logger.info("Filtering by bulkiness")
    if opts.core is None:
        raise RuntimeError("A core molecular geometry is needed to compute bulkiness")

    molecules["bulkiness"] = call_cat_in_parallel(molecules.smiles, opts)

    # Check if the molecules fulfill the bulkiness predicate
    bulkiness = opts.filters["bulkiness"]
    predicate = next(iter(bulkiness.keys()))
    limit = bulkiness[predicate]

    if predicate == "lower_than":
        has_pattern = molecules["bulkiness"] <= limit
    elif predicate == "greater_than":
        has_pattern = molecules["bulkiness"] >= limit

    logger.info(f"Keep molecules that have bulkiness {predicate} {limit}")

    return molecules[has_pattern]


def create_ouput_file(result_path: Path, k: int) -> Path:
    """Create path to print the resulting candidates."""
    parent = result_path / f"batch_{k}"
    parent.mkdir(exist_ok=True)
    return parent / "candidates.csv"


def start_logger(opts: Options) -> None:
    """Initial configuration of the logger."""
    version = pkg_resources.get_distribution('swan').version
    configure_logger(Path("."))
    logger.info(f"Using swan version: {version} ")
    logger.info(f"Working directory is: {opts.workdir}")


def main():
    """Parse the command line arguments to screen smiles."""
    parser = argparse.ArgumentParser(description="modeller -i input.yml")
    # configure logger
    parser.add_argument('-i', required=True,
                        help="Input file with options")
    args = parser.parse_args()

    options = validate_input(args.i)

    split_filter_in_batches(options)


if __name__ == "__main__":
    main()
