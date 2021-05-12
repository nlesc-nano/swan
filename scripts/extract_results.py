#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import pandas as pd

MAP_NAMES = {"11k": 11043, "1k": 1000, "2k": 2000, "500": 500, "5k": 5000, "7k": 7500}


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


def extract_subfolder(folder: Path, nfiles: int, nsamples: int) -> pd.DataFrame:
    """Extract results from given folder."""
    data = []
    for i in range(1, nfiles + 1):
        file_results = folder / f"{i}" / "results.md"
        data.append(read_info(file_results))

    names = read_property_names(folder / "1" / "results.md")

    values = np.stack(data, axis=1)
    means = np.mean(values, axis=1)

    return pd.DataFrame({"name": names, nsamples: means})


def extract_all_results(root: Path) -> None:
    """Extract all the rvalues from all the subfolders."""
    folders = [p for p in root.iterdir() if p.is_dir()]
    models = ("Fingerprints", "MPNN", "SE3Transformer")
    results = {}
    for directory in folders:
        nsamples = MAP_NAMES[directory.name]
        for model in models:
            workdir = directory / model / "Results"
            nresults = count_results(workdir)
            df = extract_subfolder(workdir, nresults, nsamples)
            if results.get(model) is None:
                results[model] = df
            else:
                old_df = results[model]
                results[model] = pd.merge(old_df, df, how="outer", on="name")

    new_results = process_data(results)
    plot_results_all_models(new_results)
    for name, df in new_results.items():
        plot_results_for_model(name, df)


def plot_results_all_models(results):
    # Create subplots to accomodate the results
    _, axis = plt.subplots(nrows=5, ncols=3, figsize=(20, 20), constrained_layout=True)

    # merge the models results
    columns = results["Fingerprints"].columns
    for k, c in enumerate(columns):
        data = pd.concat([df[c] for df in results.values()], axis=1)
        sns.lineplot(data=data, ax=axis[k // 3][k % 3])
    plt.savefig("scaling.png")


def plot_results_for_model(name: str, df: pd.DataFrame):
    # Create subplots to accomodate the results
    _, axis = plt.subplots(nrows=5, ncols=3, figsize=(20, 20), constrained_layout=True)

    # merge the models results
    for k, c in enumerate(df.columns):
        sns.lineplot(data=df[c], ax=axis[k // 3][k % 3])
    plt.suptitle(f"Scaling for {name}", fontsize=14)        
    plt.savefig(f"{name}_scaling.png")


def process_data(results):
    new = {}
    for k, df in results.items():
        df = df.set_index('name').T
        df.index = df.index.astype(int)
        df.sort_index(inplace=True)
        new[k] = df
        df.to_csv(f"{k}.csv")

    return new


def count_results(path: Path) -> int:
    """Count the number of result folders in a given path."""
    return len(list(path.iterdir()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="folder to process", required=True, type=Path)
    args = parser.parse_args()
    extract_all_results(args.folder)


if __name__ == "__main__":
    main()

