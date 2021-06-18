import numpy as np
import torch

from swan.dataset import FingerprintsData, split_dataset, load_split_dataset
from swan.modeller import GPModeller
from swan.modeller.models import GaussianProcess

from .utils_test import PATH_TEST

torch.set_default_dtype(torch.float32)


def test_train_gaussian_processes():
    """Test the training of Gaussian Processes."""
    data = FingerprintsData(PATH_TEST / "thousand.csv", properties=["Hardness (eta)"])
    # Split the data into training and validation set
    partition = split_dataset(data.fingerprints, data.labels, frac=(0.8, 0.2))

    # Model
    model = GaussianProcess(partition.features_trainset, partition.labels_trainset.flatten())

    researcher = GPModeller(model, data, use_cuda=False, replace_state=True)
    researcher.set_optimizer("Adam", lr=0.1)
    researcher.set_scheduler("StepLR", 0.1)
    researcher.data.scale_labels()
    researcher.train_model(5, partition)
    multivariate, _ = researcher.validate_model()

    assert not np.isnan(multivariate.mean).all()
    assert not np.isnan(multivariate.lower).all()
    assert not np.isnan(multivariate.upper).all()


def test_predcit_gaussian_processes():
    """Test the prediction functionality of Gaussian Processes."""
    partition = load_split_dataset()
    features, labels = [torch.from_numpy(getattr(partition, x).astype(np.float32)) for x in ("features_trainset", "labels_trainset")]
    model = GaussianProcess(features, labels.flatten())

    data = FingerprintsData(PATH_TEST / "smiles.csv", properties=None, sanitize=False)
    researcher = GPModeller(model, data, use_cuda=False, replace_state=False)
    fingers = data.fingerprints
    predicted = researcher.predict(fingers)
    assert not np.isnan(predicted.mean).all()
