import numpy as np
import torch

from swan.dataset import FingerprintsData, split_dataset
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

    researcher = GPModeller(model, data, use_cuda=False, replace_state=False)
    researcher.set_optimizer("Adam", lr=0.1)
    researcher.set_scheduler("StepLR", 0.1)
    researcher.data.scale_labels()
    researcher.train_model(5, partition)
    multivariate, _ = researcher.validate_model()

    assert not np.isnan(multivariate.mean).all()
    assert not np.isnan(multivariate.lower).all()
    assert not np.isnan(multivariate.upper).all()

    # Predict
    fingers = FingerprintsData(PATH_TEST / "smiles.csv", properties=None, sanitize=False)
    predicted = researcher.predict(fingers.fingerprints)
    assert not np.isnan(predicted.mean).all()
