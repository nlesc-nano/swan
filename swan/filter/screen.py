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
from numbers import Real
from pathlib import Path
from typing import List

import pandas as pd
import yaml
from rdkit import Chem
from schema import Optional, Or, Schema, SchemaError

from ..utils import Options
from ..features.featurizer import create_molecules

#: Schema to validate the ordering keywords
SCHEMA_ORDERING = Or(
    Schema({"greater_than": Real}),
    Schema({"lower_than": Real}),
    Schema({"equal": Real})
    )

#: Schema to validate the filters to apply for screening
SCHEMA_FILTERS = Schema({
    # Select or or more functional group using smile
    Optional("functional_groups"): Schema([str]),

    # Select smiles >, < or = to some value
    Optional("bulkiness"): SCHEMA_ORDERING,
    Optional("SA_score"): SCHEMA_ORDERING
})

#: Schema to validate the input for screening
SCHEMA_SCREEN = Schema({
    # Load the dataset from a file
    "smiles_file": str,

    # Constrains to filter
    "filters": SCHEMA_FILTERS
})


def read_molecules(input_file: Path) -> pd.DataFrame:
    """Read smiles from a csv-like file."""
    return pd.read_csv(input_file, index_col=0).reset_index(drop=True)


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


def apply_filter(opts: Options) -> None:
    """Apply a set of filters to the given smiles."""
    # Read molecules into a pandas dataframe
    molecules = read_molecules(opts.smiles_file)

    # Create a new column that will contain the label of the screened candidates
    molecules["is_candidate"] = True

    # Create rdkit representations
    molecules["rdkit_molecules"] = create_molecules(molecules['smiles'].to_numpy())

    # Apply all the filters
    available_filters = {
        "functional_groups": filter_by_functional_group}

    for key in opts.filters.keys():
        if key in available_filters:
            available_filters[key](molecules, opts)
    
    # write candidates to file



def filter_by_functional_group(molecules: pd.DataFrame, functional_groups: List[str], target: str = "candidates") -> None:
    """Search for a set of functional_groups."""
    # Transform functional_groups to rkdit molecules
    patterns = tuple((Chem.MolFromSmiles(f) for f in functional_groups))

    # Get the index of the potential candidates
    ids = molecules.index[molecules["is_candidate"]]

    # Check if the functional_groups are in the molecules
    molecules['is_candidate'] = molecules["rdkit_molecules"].apply(
        lambda m: False if m.name not in ids else any(m.HasSubstructMatch(p) for p in patterns))


def main():
    """Parse the command line arguments to screen smiles."""
    parser = argparse.ArgumentParser(description="modeller -i input.yml")
    # configure logger
    parser.add_argument('-i', required=True,
                        help="Input file with options")
    args = parser.parse_args()

    options = validate_input(args.i)

    apply_filter(options)


if __name__ == "__main__":
    main()
