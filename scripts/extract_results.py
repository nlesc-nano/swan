#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def read_info(file_name: Path, column: str = "rvalue") -> np.ndarray:
    """Read the rvalues from the table."""
    with open(file_name, 'r') as f:
        data = f.readlines()
    
    # Indices of the columns with the regression info
    names = {"slope": 0, "intercept": 1, "rvalue": 2, "stderr": 3}
    index = names[column]

    regression = [x.split()[-4:] for x in data[1:]]
    return np.array([val[index] for val in regression], float)


def read_property_names(file_name: Path, column: str = "rvalue") -> List[str]:
    """Read the rvalues from the table."""
    with open(file_name, 'r') as f:
        data = f.readlines()

    return [" ".join(x.split()[:-4]) for x in data[1:]]


def extract_results(folder: Path, nfiles: int) -> None:
    """Extract results from given folder."""
    data = []
    for i in range(1, nfiles + 1):
        file_results = folder / f"{i}" / "results.md"
        data.append(read_info(file_results))

    names = read_property_names(folder / "1" / "results.md")

    values = np.stack(data, axis=1)
    means = np.mean(values, axis=1)

    df = pd.DataFrame({"name": names, "mean": means})
    df.to_csv("means.csv", index=False, float_format="%.2f")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="folder to process", required=True, type=Path)
    parser.add_argument("-n", "--number", help="Number of copies in folder", required=True, type=int)
    args = parser.parse_args()
    extract_results(args.folder, args.number)


if __name__ == "__main__":
    main()

