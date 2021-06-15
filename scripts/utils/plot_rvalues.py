#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

MODELS = ("gaussian_process", "decision_tree", "svm", "Fingerprints", "MPNN", "SE3Transformer")
NSAMPLES = ("11k", "1k", "2k", "500", "5k", "7k")
MAP_NAMES = {"11k": 11043, "1k": 1000, "2k": 2000, "500": 500, "5k": 5000, "7k": 7500}
MSE_FILE = "MSE.json"


def read_json(file_name: str):
    with open(file_name, 'r') as f:
        xs = json.load(f)

    return xs


def read_data():
    return {n: read_json(f"All_models_{n}.json") for n in NSAMPLES}


def plot_data():
    data = read_data()
    properties = transpose_data(data)

    _, axis = plt.subplots(nrows=5, ncols=3, figsize=(20, 20), constrained_layout=True)
    for k, (name, values) in enumerate(properties.items()):
        df = pd.DataFrame(values).T
        df.sort_index(inplace=True)
        df = df ** 2

        ax = axis[k // 3][k % 3]
        obj = sns.lineplot(data=df, markers=True, ax=ax)
        obj.set_title(name)

    plt.savefig("all_properties.png")


def transpose_data(data):
    names = data[NSAMPLES[0]].keys()
    results = {n: {} for n in names}
    for n, xs in data.items():
        number = MAP_NAMES[n]
        for name, val in xs.items():
            results[name][number] = val

    return results


def main():
    plot_data()


if __name__ == "__main__":
    main()
