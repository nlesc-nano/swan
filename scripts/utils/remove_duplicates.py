#!/usr/bin/env python

import argparse
import json

import pandas as pd
from pathlib import Path
from typing import List
from itertools import chain


def read_geometries(file_name: Path) -> List[str]:
    """Read the geometries from ``file_name``."""
    with open(file_name, 'r') as f:
        gs = json.load(f)

    return gs


def remove_duplicates(folders: List[str]):
    """Remove all duplicate smiles from the given folders."""
    paths = [Path(f) for f in folders]
    smiles = pd.concat([pd.read_csv(next(p.glob("*csv")), index_col=0) for p in paths])
    smiles.reset_index(drop=True, inplace=True)
    gs = list(chain(*[read_geometries(next(p.glob("*json"))) for p in paths]))
    geometries = {k: v for k, v in enumerate(gs)}

    # remove duplicate
    new_smiles = smiles.drop_duplicates(subset=["smiles"])
    indices = new_smiles.index.to_list()
    new_smiles.reset_index(drop=True, inplace=True)
    new_geometries = [geometries[i] for i in indices]

    new_smiles.to_csv("all_carboxylics.csv")
    with open("all_geometries_carboxylics.json", 'w') as f:
        json.dump(new_geometries, f)

    print(new_geometries[-1])
    print(new_smiles.iloc[-1])


def main():
    parser = argparse.ArgumentParser()
    # configure logger
    parser.add_argument("-f", "--folders", help="folders to look for the data", nargs="+")
    args = parser.parse_args()
    remove_duplicates(args.folders)


if __name__ == "__main__":
    main()
