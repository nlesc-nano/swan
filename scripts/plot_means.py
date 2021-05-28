#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from pathlib import Path

MODELS = ("Fingerprints", "MPNN", "SE3Transformer")
NSAMPLES = ("11k", "1k", "2k", "500", "5k", "7k")
MAP_NAMES = {"11k": 11043, "1k": 1000, "2k": 2000, "500": 500, "5k": 5000, "7k": 7500}
PATH_GROUND_TRUE = Path("data/Carboxylic_acids/CDFT/cdft_random_500.csv")
MSE_FILE = "MSE.json"


def read_data():
    results = {}
    for m in MODELS:
        results[m] = {}
        for n in NSAMPLES:
            df = pd.read_json(f"means_{m}_{n}.json")
            results[m][n] = df.sum().to_dict()

    data = [pd.DataFrame(transpose_data(results[m])) for m in MODELS]
    for df in data:
        df.sort_index(inplace=True)
    for df, model in zip(data, MODELS):
        df.to_csv(f"{model}.csv")


def plot_data(model: str):
    with open(MSE_FILE, 'r') as f:
        data = json.load(f)

    data = [pd.DataFrame(transpose_data(data[m])) for m in MODELS]
    for df in data:
        df.sort_index(inplace=True)

    names = data[0].columns
    _, axis = plt.subplots(nrows=5, ncols=3, figsize=(20, 20), constrained_layout=True)

    for k, prop in enumerate(names):
        ax = axis[k // 3][k % 3]
        frames = pd.DataFrame({m: df[names[k]] for m, df in zip(MODELS, data)})
        obj = sns.lineplot(data=frames, markers=True, ax=ax)
        obj.set_title(prop)

    plt.savefig("MSE.png")


def transpose_data(data):
    names = data[NSAMPLES[0]].keys()
    results = {n: {} for n in names}
    for n, xs in data.items():
        number = MAP_NAMES[n]
        for name, val in xs.items():
            results[name][number] = val

    return results


def main():
    read_data()
    # plot_data(MODELS[0])


if __name__ == "__main__":
    main()
