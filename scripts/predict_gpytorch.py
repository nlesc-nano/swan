#!/usr/bin/env python

from pathlib import Path

import torch

from swan.dataset import FingerprintsData
from swan.modeller import GPModeller
from swan.modeller.models import GaussianProcess

path_data = Path("tests/files/smiles.csv")


def main():
    data = FingerprintsData(path_data, properties=None, sanitize=False)
    fingers = data.fingerprints
    model = GaussianProcess(fingers, torch.empty(fingers.shape[0]))
    researcher = GPModeller(model, data, use_cuda=False, replace_state=False)
    researcher.load_model("swan_chk.pt")
    data.load_scale()
    transfomer = data.transformer
    print(transfomer.center_, transfomer.scale_)
    prediction = researcher.predict(data.fingerprints)
    print(prediction.mean)
    # print(transfomer.inverse_transform(prediction.mean.numpy().reshape(-1, 1)))


if __name__ == "__main__":
    main()
