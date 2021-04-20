#!/usr/bin/env python

import argparse
import json
from collections import OrderedDict
from typing import List

import pandas as pd


def remove_elements(file_csv: str, file_geometries: str, elements: List[str]) -> None:
    """Remove them smiles and geometries containing ``elements``."""
    df = pd.read_csv(file_csv, index_col=0)
    gs = OrderedDict({k: v for k, v in enumerate(read_geometries(file_geometries))})
    for elem in elements:
        indices_to_drop = df[df.smiles.str.contains(elem)].index
        df.drop(indices_to_drop, inplace=True)
        for i in indices_to_drop:
            gs.pop(i)

    # Store new smiles
    df.to_csv("new_smiles.csv")

    # store new geometries
    with open("new_geometries.json", 'w') as f:
        json.dump(list(gs.values()), f)


def read_geometries(file_geometries: str) -> List[str]:
    """Read a list of geometries."""
    with open(file_geometries, 'r') as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser()
    # configure logger
    parser.add_argument("-c", "--csv", help="CSV file with the smiles", required=True)
    parser.add_argument("-g", "--geometry", help="Geometry file", required=True)
    parser.add_argument("-e", "--elements", help="Elements to remove", required=True, nargs='+')
    args = parser.parse_args()
    remove_elements(args.csv, args.geometry, args.elements)


if __name__ == "__main__":
    main()
