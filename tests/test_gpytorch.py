import torch

from swan.dataset import FingerprintsData, split_dataset
from swan.modeller import GPModeller
from swan.modeller.models import GaussianProcess

from .utils_test import PATH_TEST

torch.set_default_dtype(torch.float32)


def test_gaussian_process():
    """Test the training of Gaussian Processes."""
    torch.set_default_dtype(torch.float32)
    data = FingerprintsData(PATH_TEST / "thousand.csv", properties=["Hardness (eta)"])
    # Split the data into training and validation set
    partition = split_dataset(data.fingerprints, data.labels, frac=(0.8, 0.2))

    # Model
    model = GaussianProcess(partition.features_trainset, partition.labels_trainset.flatten())

    researcher = GPModeller(model, data, use_cuda=False, replace_state=False)
    researcher.set_optimizer("Adam", lr=0.1)
    researcher.set_scheduler("StepLR", 0.1)
    researcher.data.scale_labels()
    multivariate, _ = researcher.train_model(5, partition)

    lower, upper = multivariate.confidence_region()
    assert not multivariate.mean.isnan().all().item()
    assert not lower.isnan().all().item()
    assert not upper.isnan().all().item()
