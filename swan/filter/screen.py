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
from functools import partial
from numbers import Real
from pathlib import Path
from typing import FrozenSet

import h5py
import numpy as np
import pandas as pd
import tempfile
import yaml

from dataCAT import prop_to_dataframe
from rdkit import Chem
from schema import Optional, Or, Schema, SchemaError

from ..utils import Options
from ..cosmo.cat_interface import call_cat


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

    # File to print the final candidates
    Optional("output_file", default="candidates.csv"): str
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


def apply_filters(opts: Options) -> None:
    """Apply a set of filters to the given smiles."""
    # Read molecules into a pandas dataframe
    molecules = read_molecules(opts.smiles_file)

    # Create rdkit representations
    converter = np.vectorize(Chem.MolFromSmiles)
    back_converter = np.vectorize(Chem.MolToSmiles)
    molecules["rdkit_molecules"] = converter(molecules.smiles)

    # Convert smiles to the standard representation
    mols = molecules.rdkit_molecules
    molecules.smiles = back_converter(mols[mols.notnull()])

    # Create a new column that will contain the labels of the screened candidates
    molecules["is_candidate"] = mols.notnull()

    # Apply all the filters
    available_filters = {
        "include_functional_groups": include_functional_groups,
        "exclude_functional_groups": exclude_functional_groups,
        "bulkiness": filter_by_bulkiness}

    for key in opts.filters.keys():
        if key in available_filters:
            molecules = available_filters[key](molecules, opts)

    molecules.to_csv(opts.output_file, columns=["smiles"])
    print(f"The filtered candidates has been written to the {opts.output_file} file!")


def include_functional_groups(molecules: pd.DataFrame, opts: Options) -> pd.DataFrame:
    """Check that the molecules contain some functional groups."""
    return filter_by_functional_group(molecules, opts, "include_functional_groups", False)


def exclude_functional_groups(molecules: pd.DataFrame, opts: Options) -> pd.DataFrame:
    """Check that the molecules do not contain some functional groups."""
    return filter_by_functional_group(molecules, opts, "exclude_functional_groups", True)


def filter_by_functional_group(molecules: pd.DataFrame, opts: Options, key: str,
                               exclude: bool) -> pd.DataFrame:
    """Search for a set of functional_groups."""
    # Transform functional_groups to rkdit molecules
    functional_groups = opts["filters"][key]
    patterns = {Chem.MolFromSmiles(f) for f in functional_groups}

    # Create rdkit representations
    converter = np.vectorize(Chem.MolFromSmiles)
    # back_converter = np.vectorize(Chem.MolToSmiles)
    molecules["rdkit_molecules"] = converter(molecules.smiles)

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
    if opts.core is None:
        raise RuntimeError("A core molecular geometry is needed to compute bulkiness")

    # compute bulkiness using CAT
    molecules["bulkiness"] = compute_bulkiness(molecules, opts)

    # Check if the molecules fulfill the bulkiness predicate
    bulkiness = opts.filters["bulkiness"]
    predicate = next(iter(bulkiness.keys()))
    value = bulkiness[predicate]

    if predicate == "lower_than":
        has_pattern = molecules["bulkiness"] <= value
    elif predicate == "greater_than":
        has_pattern = molecules["bulkiness"] >= value

    return molecules[has_pattern]


def compute_bulkiness(molecules: pd.DataFrame, opts: Options) -> pd.Series:
    """Compute the bulkiness for the candidates."""
    path_hdf5 = call_cat(molecules, opts)
    with h5py.File(path_hdf5, 'r') as f:
        dset = f['qd/properties/V_bulk']
        df = prop_to_dataframe(dset)

    # flat the dataframe and remove duplicates
    df = df.reset_index()

    # make anchor atom neutral to compare with the original
    # TODO make it more general
    df.ligand = df.ligand.str.replace("[O-]", "O", regex=False)

    # remove duplicates
    df.drop_duplicates(subset=['ligand'], keep='first', inplace=True)

    # Extract the bulkiness
    bulkiness = pd.merge(molecules, df, left_on="smiles", right_on="ligand")["V_bulk"]

    if len(molecules.index) != len(bulkiness):
        msg = "There is an incongruence in the bulkiness computed by CAT!"
        raise RuntimeError(msg)

    return bulkiness.to_numpy()


def main():
    """Parse the command line arguments to screen smiles."""
    parser = argparse.ArgumentParser(description="modeller -i input.yml")
    # configure logger
    parser.add_argument('-i', required=True,
                        help="Input file with options")
    args = parser.parse_args()

    options = validate_input(args.i)

    apply_filters(options)


if __name__ == "__main__":
    main()
