#!/usr/bin/env python

import argparse

import pandas as pd


def extract_data(file_name: str, name: str = "cdft") -> None:
    """Get the smiles plus their properties."""
    data = pd.read_csv(file_name, header=[0, 1], index_col=0)[name]
    data.reset_index(inplace=True)
    data.drop([0, 1], inplace=True)
    data.dropna(inplace=True)
    data.rename(columns={'index': 'smiles'}, inplace=True)
    data.reset_index(inplace=True, drop=True)
    data.to_csv("ouput.csv")


def main():
    parser = argparse.ArgumentParser()
    # configure logger
    parser.add_argument("-c", "--csv", help="CSV files with the smiles")
    parser.add_argument("-n", "--name", help="N")
    args = parser.parse_args()
    extract_data(args.csv)


if __name__ == "__main__":
    main()
